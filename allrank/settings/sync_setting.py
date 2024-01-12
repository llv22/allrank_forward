import os
import argparse
import json
from pathlib import Path
import copy

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=str, default="allrank/settings/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth.json")
    conf = args.parse_args()
    
    with open(conf.input, "r") as f:
        standard = json.load(f)
    
    target_f = [[1], [2], [3], [4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17], [18]]
    parent_folder = Path(conf.input).parent
    short_name = Path(conf.input).stem
    folder = conf.input.replace('allrank/settings/', '').replace('.json', '')
    run_files = [f"PYTHONPATH=.:${{PYTHONPATH}} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name {conf.input} --run-id {folder} --job-dir experiments/{folder}"]
    for f in target_f:
        dic = copy.deepcopy(standard)
        dic['data']['mark_feature_indexes'] = [i -1 for i in f]
        name_postfix = "_".join([str(i) for i in f])
        feature_file = os.path.join(parent_folder, f"{short_name}_without_feature{name_postfix}.json")
        with open(feature_file, "w") as f:
            json.dump(dic, f, indent=4)
        folder = feature_file.replace('allrank/settings/', '').replace('.json', '')
        run_files.append(f"PYTHONPATH=.:${{PYTHONPATH}} CUDA_VISIBLE_DEVICES=0 python allrank/main.py  --config-file-name {feature_file} --run-id {folder} --job-dir experiments/{folder}")
    
    with open(f"{parent_folder}/run_{short_name}_group.sh", "w") as f:
        f.write("\n".join(run_files))