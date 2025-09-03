import os
import glob
import json
import sys

def process_cluster(cluster_idx):
    input_dir = f'output_cluster_{cluster_idx}'
    output_dir = '../stage1_filtered'
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(input_dir, '*.tokenized.jsonl')
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"No tokenized jsonl files found in {input_dir}")
        return

    instances = []
    for f in file_list:
        with open(f, 'r', encoding='utf-8') as infile:
            for line in infile:
                item = json.loads(line)
                instances.append(item)
    final_obj = {
        "type": "text_only",
        "instances": instances
    }

    # Build output filename correctly!
    out_file = os.path.join(output_dir, f"cluster{cluster_idx}_stage1_filtered.json")
    with open(out_file, 'w', encoding='utf-8') as out:
        json.dump(final_obj, out, ensure_ascii=False, indent=2)
    print(f"Done! Written: {out_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_stage1_postprocess.py <cluster_idx>")
        sys.exit(1)
    # Use the *second* argument (the first after the script name)
    cluster_idx = sys.argv[1]
    process_cluster(cluster_idx)

