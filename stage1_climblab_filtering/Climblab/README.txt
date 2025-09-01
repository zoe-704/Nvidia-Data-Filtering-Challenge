1) download climblab clusters (1, 2, 6, 8, 11, 12, 18), all parquet files
2) run 'bash run_stage1.sh cluster_<1,2,6,8,11,12,18>' for each cluster, fltered parquet to json files are in output directory output_cluster_<1,2,6,8,11,12,18>
3) in each output_cluster_<> run the following command:

jq -s ‘.’ *.tokenized.jsonl > ../../stage1_filtered/cluster<>_stage1_filtered.json
