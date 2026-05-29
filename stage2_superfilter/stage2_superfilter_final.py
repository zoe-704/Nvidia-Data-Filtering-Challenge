#!/usr/bin/env python3
"""
Final Data Curation Script - SuperFilter v2.0

This script performs final data curation by:
1. Calculating perplexity scores using dual models for robust quality assessment
2. Creating a mixed dataset from the top 75% with the top 25% duplicated
3. Shuffling the final dataset for training diversity
4. Outputting clean data without internal scores for release

Features:
- Dual model perplexity scoring with optimized tokenization
- Mixed sampling strategy for balanced quality/diversity
- Clean output format for production use
- Error handling and logging

Version: 2.0 (Final Release)
"""

import os
import json
import glob
import random
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import gc
from tqdm import tqdm


class SuperFilterFinal:
    """
    Final data curation system using dual model perplexity scoring.
    
    This class implements a sophisticated filtering pipeline that:
    - Uses two models for robust quality assessment
    - Creates mixed datasets for optimal training balance
    - Outputs production-ready clean data
    """
    
    def __init__(self, 
                 model_path_1: str, 
                 model_path_2: str, 
                 input_dir: str, 
                 output_dir: str,
                 batch_size: int = 40,
                 random_seed: int = 42):
        """
        Initialize the SuperFilter system.
        
        Args:
            model_path_1: Path to the first model
            model_path_2: Path to the second model
            input_dir: Directory containing input JSON files
            output_dir: Directory to save final outputs
            batch_size: Batch size for processing
            random_seed: Random seed for reproducible shuffling
        """
        self.model_path_1 = model_path_1
        self.model_path_2 = model_path_2
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # Initialize components
        self.tokenizer = None
        self.model_1 = None
        self.model_2 = None
        self.records = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_models(self) -> None:
        """Load the shared tokenizer and both models."""
        print("Loading SuperFilter components...")
        print("Loading shared tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_1)
        
        print("Loading Model 1...")
        self.model_1 = AutoModelForCausalLM.from_pretrained(self.model_path_1).to(self.device)
        self.model_1.eval()
        
        print("Loading Model 2...")
        self.model_2 = AutoModelForCausalLM.from_pretrained(self.model_path_2).to(self.device)
        self.model_2.eval()
        
        print("SuperFilter models loaded successfully!")
        print(f"   Model 1: {self.model_path_1}")
        print(f"   Model 2: {self.model_path_2}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
    
    def load_data(self) -> None:
        """Load all records from JSON files in the input directory."""
        print("\nLoading input data...")
        self.records = []
        
        json_files = glob.glob(os.path.join(self.input_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.input_dir}")
        
        for file_path in tqdm(json_files, desc="Loading files", unit="file"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    instances = data.get("instances", [])
                    self.records.extend(instances)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        if not self.records:
            raise ValueError("No records loaded from input files")
        
        print(f"Loaded {len(self.records):,} samples from {len(json_files)} files")
    
    def compute_perplexity_scores_single_model(self, inputs, model) -> np.ndarray:
        """
        Compute perplexity scores using a single model with pre-tokenized inputs.
        
        Args:
            inputs: Pre-tokenized inputs (already on device)
            model: The model to use for scoring
            
        Returns:
            Array of perplexity scores
        """
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            attention = inputs["attention_mask"][..., 1:].contiguous()
            sentence_loss = (loss * attention).sum(dim=1) / attention.sum(dim=1)
            scores = -sentence_loss.cpu().numpy()
        
        return scores
    
    def compute_dual_perplexity_scores(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Compute perplexity scores using both models and average them.
        Optimized to tokenize only once and reuse tokens for both models.
        Also returns token counts for statistics tracking.
        
        Args:
            texts: List of text strings to score
            
        Returns:
            Tuple of (averaged perplexity scores, token counts per text)
        """
        # Tokenize once for both models (shared tokenizer)
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        # Extract token counts (excluding padding tokens)
        token_counts = []
        for i, attention_mask in enumerate(inputs["attention_mask"]):
            # Count non-padding tokens (where attention_mask == 1)
            token_count = attention_mask.sum().item()
            token_counts.append(token_count)
        
        # Get scores from both models using the same tokenized inputs
        scores_1 = self.compute_perplexity_scores_single_model(inputs, self.model_1)
        scores_2 = self.compute_perplexity_scores_single_model(inputs, self.model_2)
        
        # Average the scores for robust assessment
        averaged_scores = (scores_1 + scores_2) / 2.0
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return averaged_scores, token_counts
    
    def calculate_scores(self) -> None:
        """Calculate perplexity scores for all records using dual model approach and track token counts."""
        print("\nComputing dual model perplexity scores and tracking tokens...")
        total_samples = len(self.records)
        
        with tqdm(total=total_samples, desc="Processing samples", unit="samples", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for i in range(0, total_samples, self.batch_size):
                batch = self.records[i:i+self.batch_size]
                texts = [rec.get("text", "") for rec in batch]
                
                try:
                    scores, token_counts = self.compute_dual_perplexity_scores(texts)
                    for rec, score, token_count in zip(batch, scores, token_counts):
                        rec["perplexity_score"] = float(score)
                        rec["token_count"] = int(token_count)  # Store token count from scoring
                except Exception as e:
                    print(f"\nWarning: Failed to process batch {i//self.batch_size + 1}: {e}")
                    # Assign default values for failed batches
                    for rec in batch:
                        rec["perplexity_score"] = 0.0
                        rec["token_count"] = 0
                
                pbar.update(len(batch))
        
        print("Dual model scoring and token tracking completed successfully!")
    
    @staticmethod
    def _quality_score(record: Dict[str, Any]) -> float:
        """Return the score used for ranking."""
        return record.get("perplexity_score", 0.0)
    
    def create_final_dataset(self) -> List[Dict[str, Any]]:
        """
        Create the final mixed dataset with top 25% duplicated for emphasis.
        
        Strategy: Take top 75% + duplicate the top 25% (so top 25% appears twice)
        This gives higher quality samples more weight in training.
        
        Returns:
            List of final dataset records (shuffled, without scores)
        """
        print("\nCreating final mixed dataset with top 25% duplication...")
        
        # Sort records by perplexity score (descending - higher scores are better)
        records_sorted = sorted(self.records, key=self._quality_score, reverse=True)
        total_records = len(records_sorted)
        
        # Calculate thresholds
        top_25_count = int(total_records * 0.25)
        top_75_count = int(total_records * 0.75)
        
        print("Dataset composition strategy:")
        print(f"   Top 25% samples: {top_25_count:,} ")
        print(f"   Top 75% samples: {top_75_count:,}")
        
        # Get top 25% and top 75% samples
        top_25_samples = records_sorted[:top_25_count]
        top_75_samples = records_sorted[:top_75_count]
        
        # Create mixed dataset: top 75% + duplicated top 25%
        final_samples = top_75_samples.copy()  # Start with top 75%
        final_samples.extend(top_25_samples)   # Add top 25% again (duplication)
        
        print("Final dataset composition:")
        print(f"   Top 75% samples: {top_75_count:,}")
        print(f"   + Duplicated top 25%: {top_25_count:,}")
        print(f"   = Total final samples: {len(final_samples):,}")
        print(f"   Quality boost: Top 25% appears 2x for enhanced training")

        # Shuffle for training diversity (mixes original and duplicated samples)
        print("Shuffling dataset for optimal training diversity...")
        random.shuffle(final_samples)
        
        print("Final mixed dataset created and shuffled!")
        print("High-quality samples (top 25%) now have 2x training weight")
        return final_samples  # Return with metadata for token stats computation
    
    def compute_final_token_stats(self, final_samples_with_metadata: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Compute token statistics from the final dataset using pre-computed token counts.
        This avoids re-tokenizing by using token counts from the scoring phase.
        
        Args:
            final_samples_with_metadata: Final samples that still contain token_count metadata
            
        Returns:
            Dictionary with token statistics
        """
        print("Computing token statistics from perplexity scoring data...")
        
        total_tokens = 0
        total_chars = 0
        sample_count = 0
        
        for record in final_samples_with_metadata:
            # Use pre-computed token count from scoring
            token_count = record.get("token_count", 0)
            text = record.get("text", "")
            
            if token_count > 0 and text:
                total_tokens += token_count
                total_chars += len(text)
                sample_count += 1
        
        token_stats = {
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "average_tokens_per_sample": round(total_tokens / sample_count, 2) if sample_count > 0 else 0,
            "average_chars_per_sample": round(total_chars / sample_count, 2) if sample_count > 0 else 0
        }
        
        print("Token statistics computed (no re-tokenization needed):")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Avg tokens/sample: {token_stats['average_tokens_per_sample']:,}")
        print(f"   Avg chars/sample: {token_stats['average_chars_per_sample']:,}")
        
        return token_stats


    def save_final_output(self, final_dataset_with_metadata: List[Dict[str, Any]]) -> None:
        """
        Save the final curated dataset and metadata in separate files.
        
        Args:
            final_dataset_with_metadata: The final dataset still containing metadata (scores, token counts)
        """
        print("\nSaving final curated dataset...")
        
        # Compute token statistics using pre-computed token counts from scoring
        token_stats = self.compute_final_token_stats(final_dataset_with_metadata)
        
        # Clean the dataset for release (remove scores and token counts)
        final_dataset = []
        for record in final_dataset_with_metadata:
            clean_record = {
                k: v for k, v in record.items()
                if k not in ["perplexity_score", "token_count"]
            }
            final_dataset.append(clean_record)
        
        # Create metadata with token statistics
        metadata = {
            "total_samples": len(final_dataset),
            "total_tokens": token_stats["total_tokens"],
            "total_characters": token_stats["total_characters"],
            "average_tokens_per_sample": token_stats["average_tokens_per_sample"],
            "average_chars_per_sample": token_stats["average_chars_per_sample"],
            "curation_method": "dual_model_perplexity_superfilter",
            "quality_weighting": "2x_weight_for_highest_quality_samples",
            "models_used": [
                os.path.basename(self.model_path_1),
                os.path.basename(self.model_path_2)
            ],
            "shuffled": True,
            "version": "2.0",
            "created_by": "SuperFilter v2.0",
            "input_directory": self.input_dir,
            "batch_size": self.batch_size
        }
        
        # Create the final dataset structure (clean, no metadata)
        output_data = {
            "type": "text_only",
            "instances": final_dataset
        }
        
        # Define output paths
        final_output_path = os.path.join(self.output_dir, "final_curated_dataset.json")
        metadata_output_path = os.path.join(self.output_dir, "curation_metadata.txt")
        
        try:
            # Save the final dataset (clean)
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Save the metadata separately
            with open(metadata_output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print("Final dataset and metadata saved successfully!")
            print(f"   Dataset file: {final_output_path}")
            print(f"   Metadata file: {metadata_output_path}")
            print(f"   Total samples: {len(final_dataset):,}")
            print(f"   Total tokens: {token_stats['total_tokens']:,}")
            print(f"   Total characters: {token_stats['total_characters']:,}")
            print("   Quality: Top 75% + duplicated top 25%")
            print("   Shuffled: Yes (for training diversity)")
            
        except Exception as e:
            print(f"Error saving files: {e}")
            raise
    
    def run(self) -> None:
        """Execute the complete SuperFilter pipeline."""
        print("Starting SuperFilter - Final Data Curation")
        print("=" * 60)
        
        try:
            # Load models and data
            self.load_models()
            self.load_data()
            
            # Calculate perplexity scores
            self.calculate_scores()
            
            # Create final mixed dataset (with metadata for token stats)
            final_dataset_with_metadata = self.create_final_dataset()
            
            # Save final output (will clean the dataset internally)
            self.save_final_output(final_dataset_with_metadata)
            
            print("\n" + "=" * 60)
            print("SuperFilter completed successfully!")
            print("High-quality curated dataset ready for training!")
            
        except Exception as e:
            print(f"\nSuperFilter failed: {e}")
            raise
        
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Memory cleanup completed")


def main():
    """Main function to run the SuperFilter final curation."""
    # Configuration for final release
    config = {
        "model_path_1": "./llama400m_ft_exp_fc",
        "model_path_2": "./llama400m_ft_exp_re",
        "input_dir": "../stage1_climblab_filtering/stage1_filtered/",
        "output_dir": "../final_curated_dataset/",
        "batch_size": 40
    }
    
    # Create and run the SuperFilter
    superfilter = SuperFilterFinal(**config)
    superfilter.run()


if __name__ == "__main__":
    main()
    
