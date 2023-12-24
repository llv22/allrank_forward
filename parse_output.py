import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input string")
    args = parser.parse_args()

    # Read input file
    input = args.input
    input = input[input.find("Epoch : 99 ")+1:]
    train_loss = input[input.find("Train loss:") + len("Train loss:"):input.find("Val loss:")].strip()
    val_loss = input[input.find("Val loss:") + len("Val loss:"):input.find("Train ndcg_1")].strip()
    train_ndcg_1 = input[input.find("Train ndcg_1") + len("Train ndcg_1"):input.find("Train ndcg_5")].strip()
    train_ndcg_5 = input[input.find("Train ndcg_5") + len("Train ndcg_5"):input.find("Train ndcg_10")].strip()
    train_ndcg_10 = input[input.find("Train ndcg_10") + len("Train ndcg_10"):input.find("Train mrr_1")].strip()
    train_mrr_1 = input[input.find("Train mrr_1") + len("Train mrr_1"):input.find("Train mrr_5")].strip()
    train_mrr_5 = input[input.find("Train mrr_5") + len("Train mrr_5"):input.find("Train mrr_10")].strip()
    train_mrr_10 = input[input.find("Train mrr_10") + len("Train mrr_10"):input.find("Train ap_1")].strip()
    train_ap_1 = input[input.find("Train ap_1") + len("Train ap_1"):input.find("Train ap_5")].strip()
    train_ap_5 = input[input.find("Train ap_5") + len("Train ap_5"):input.find("Train ap_10")].strip()
    train_ap_10 = input[input.find("Train ap_10") + len("Train ap_10"):input.find("Val ndcg_1")].strip()
    val_ndcg_1 = input[input.find("Val ndcg_1") + len("Val ndcg_1"):input.find("Val ndcg_5")].strip()
    val_ndcg_5 = input[input.find("Val ndcg_5") + len("Val ndcg_5"):input.find("Val ndcg_10")].strip()
    val_ndcg_10 = input[input.find("Val ndcg_10") + len("Val ndcg_10"):input.find("Val mrr_1")].strip()
    val_mrr_1 = input[input.find("Val mrr_1") + len("Val mrr_1"):input.find("Val mrr_5")].strip()
    val_mrr_5 = input[input.find("Val mrr_5") + len("Val mrr_5"):input.find("Val mrr_10")].strip()
    val_mrr_10 = input[input.find("Val mrr_10") + len("Val mrr_10"):input.find("Val ap_1")].strip()
    val_ap_1 = input[input.find("Val ap_1") + len("Val ap_1"):input.find("Val ap_5")].strip()
    val_ap_5 = input[input.find("Val ap_5") + len("Val ap_5"):input.find("Val ap_10")].strip()
    val_ap_10 = input[input.find("Val ap_10") + len("Val ap_10"):].strip()
    output = f"|{float(train_loss):.6f}|{float(train_ndcg_1):.4f}|{float(train_ndcg_5):.4f}|{float(train_ndcg_10):.4f}|{float(train_mrr_1):.4f}|{float(train_mrr_5):.4f}|{float(train_mrr_10):.4f}|{float(train_ap_1):.4f}|{float(train_ap_5):.4f}|{float(train_ap_10):.4f}|{float(val_loss):.6f}|{float(val_ndcg_1):.4f}|{float(val_ndcg_5):.4f}|{float(val_ndcg_10):.4f}|{float(val_mrr_1):.4f}|{float(val_mrr_5):.4f}|{float(val_mrr_10):.4f}|{float(val_ap_1):.4f}|{float(val_ap_5):.4f}|{float(val_ap_10):.4f}|"
    print(output)