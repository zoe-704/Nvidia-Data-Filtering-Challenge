Download all parquet files for Climblab clusters <1, 2, 6, 8, 11, 12, 18> with 'python hf_download_climblab.py cluster_<>' i.e. 'python hf_download_climblab.py cluster_1’ for cluster 1 data
Run 'bash run_stage1.sh cluster_<1,2,6,8,11,12,18>' for each cluster (i.e. ‘bash run_stage1.sh cluster_1’)
Filtered parquets to json files are in output directory output_cluster_<1,2,6,8,11,12,18>
Run 'run_stage1_postprocess.py <>’ for each cluster (i.e. 'run_stage1_postprocess.py 1’) which will:
   a) Combine each parquet filtered json to a single file (i.e., cluster1_stage1_filtered.json) into ../stage1_filtered directory
   b) Format files in to LMFlow text_only type
