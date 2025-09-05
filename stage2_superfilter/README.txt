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
