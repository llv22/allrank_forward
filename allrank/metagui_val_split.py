import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_label", type=str, default="mmdataset/Feature_18_zeroshot_label2/val_qid_label2.json")
    parser.add_argument("--input", type=str, default="mmdataset/Feature_18_zeroshot_label2/val.txt")
    parser.add_argument("--output_folder", type=str, default="mmdataset/Feature_18_metagui_label2")
    args = parser.parse_args()
    
    with open(args.query_label, "r") as f:
        query_to_label = json.load(f)
    length = len(query_to_label) / 2
    cnt = 0
    val_label, test_label = {}, {}
    for k, v in query_to_label.items():
        if cnt < length:
            val_label[k] = v
            cnt += 1
        else:
            test_label[k] = v
    val_data, test_data = [], []
    with open(args.input, "r") as f:
        for l in f.readlines():
            data = l.split()
            key_to_value = {c.split(":")[0]:c.split(":")[1] for c in data[1:]}
            key_to_value['label'] = data[0]
            if int(key_to_value['qid']) in list(val_label.values()):
                val_data.append(key_to_value)
            else:
                test_data.append(key_to_value)
    print(f"val label: {len(val_label)}, val data: {len(val_data)}, test label: {len(test_label)}, test data: {len(test_data)}")
    with open(f"{args.output_folder}/val_qid_label2.json", "w") as f:
        json.dump(val_label, f, indent=2)
    with open(f"{args.output_folder}/val.txt", "w") as f:
        for d in val_data:
            f.write(d['label'] + " " + " ".join([f"{k}:{v}" for k, v in d.items() if k != 'label']) + "\n")
    with open(f"{args.output_folder}/test_qid_label2.json", "w") as f:
        json.dump(test_label, f, indent=2)
    with open(f"{args.output_folder}/test.txt", "w") as f:
        for d in test_data:
            f.write(d['label'] + " " + " ".join([f"{k}:{v}" for k, v in d.items() if k != 'label']) + "\n")