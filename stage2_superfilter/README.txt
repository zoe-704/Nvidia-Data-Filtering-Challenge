1) simply run 'python stage2_superfilter_final.py'
2) default configuration in the script for model path, input jsons and final output
3) change batch size for specific GPU capability, batch size 40 runs fine on RTX5070Ti (16GB)
    config = {
        "model_path_1": "./llama400m_ft_exp_fc",
        "model_path_2": "./llama400m_ft_exp_re",
        "input_dir": "../stage1_climblab_filtering/stage1_filtered/",
        "output_dir": "../final_curated_dataset/",
        "batch_size": 40
    }
