1) download climblab clusters (1, 2, 6, 8, 11, 12, 18), all parquet files. 'python hf_download_climblab.py cluster_1'
2) run 'bash run_stage1.sh cluster_<1,2,6,8,11,12,18>' for each cluster, fltered parquet to json files are in output directory output_cluster_<1,2,6,8,11,12,18>
3) run 'run_stage1_postprocess.py  cluster_index" where cluster_index is 1,2, 6, 8, 11, 12, 18, which will do
   a) combine each parquet filtered jsonl to a single file, i.e., cluster1_stage1_filtered.json, into ../stage1_filtered directoy
   b) format it lmflow text_only type

