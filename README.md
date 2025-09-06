# Nvidia-Data-Filtering-Challenge

1. Clone repository and pull all files from git LFS 
	
	Run 'git clone <repository-url>'

	Run 'git lfs fetch --all && git lfs checkout'

3. Download climblab data
   
	Navigate to stage1_climblab_filtering/Climblab

	Download all parquet files for Climblab clusters <1, 2, 6, 8, 11, 12, 18> with 'python hf_download_climblab.py cluster_<>' i.e. 'python hf_download_climblab.py cluster_1’ for cluster 1 data

4. Run stage 1 filtering
   
	Run 'bash run_stage1.sh cluster_<1,2,6,8,11,12,18>' for each cluster (i.e. ‘bash run_stage1.sh cluster_1’)

	Filtered parquets to json files are in output directory output_cluster_<1,2,6,8,11,12,18>

	Run 'run_stage1_postprocess.py <>’ for each cluster (i.e. 'run_stage1_postprocess.py 1’) which will:

	a. Combine each parquet filtered json to a single file (i.e., cluster1_stage1_filtered.json) into ../stage1_filtered directory

	b. Format files in to LMFlow text_only type

5. Run stage 2 filtering
   
	Run 'python stage2_superfilter_final.py'

	Default configuration in the script for model path, input json files, and final output

	Change batch size for specific GPU capability, batch size 40 runs fine on RTX5070Ti (16GB)

config = {

	"model_path_1": "./llama400m_ft_exp_fc",
 
	"model_path_2": "./llama400m_ft_exp_re",
 
	"input_dir": "../stage1_climblab_filtering/stage1_filtered/",
 
	"output_dir": "../final_curated_dataset/",
 
	"batch_size": 40
 
}

This is the instructions for submission for the Nvidia Data Filtering Challenge for Training Edge Language Models from team ZS. 
