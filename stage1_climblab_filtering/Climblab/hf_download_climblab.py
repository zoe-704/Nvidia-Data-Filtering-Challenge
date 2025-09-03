#!/usr/bin/env python3
"""
hf_download_climblab.py
Download selected ClimbLab shards from Hugging Face to a local folder, preserving cluster structure.

Example:
  python hf_download_climblab.py \
    --clusters 1 2 8 11 12 18 \
    --files-per-cluster 0 \
    --outdir ./
"""
import argparse, os
from typing import List
from huggingface_hub import list_repo_files, hf_hub_download

def list_cluster_files(repo_id: str, cluster_id: int) -> List[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = f"cluster_{cluster_id}/cluster_{cluster_id}_"
    return sorted([p for p in files if p.startswith(prefix) and p.endswith(".tokenized.parquet")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/ClimbLab")
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--files-per-cluster", type=int, default=0, help="0 = all files in cluster")
    ap.add_argument("--outdir", default="./")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    manifest_dir = os.path.join(args.outdir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    for cid in sorted(set(args.clusters)):
        paths = list_cluster_files(args.repo, cid)
        if not paths:
            print(f"[WARN] no files found for cluster {cid}")
            continue
        if args.files_per_cluster > 0:
            paths = paths[:args.files_per_cluster]

        # write manifest (repo paths)
        with open(os.path.join(manifest_dir, f"cluster_{cid}.manifest.txt"), "w") as mf:
            for rp in paths:
                mf.write(rp + "\n")

        for rp in paths:
            local_path = hf_hub_download(
                repo_id=args.repo, repo_type="dataset", filename=rp,
                local_dir=args.outdir, local_dir_use_symlinks=False
            )
            print("[OK]", local_path)

    print("[DONE] Downloaded. Base folder:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
