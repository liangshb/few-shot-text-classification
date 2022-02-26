import os
from datasets import load_from_disk, DatasetDict


def filter_pos_label(example):
    return [bool(label) for label in example["label"]]


def filter_neg_label(example):
    return [not bool(label) for label in example["label"]]


def add_len(example):
    return {
        "length": [len(tokens) for tokens in example["merged-tokens-sym"]]
    }


def main():
    dataset_path = "data/sysevr"
    split_path = "data/sysevr_splited_len"
    domains = ["API_function_call", "Arithmetic_expression", "Array_usage", "Pointer_usage"]
    for domain in domains:
        print(f"Domain: {domain}")
        dataset = load_from_disk(os.path.join(dataset_path, domain))
        dataset = dataset.map(add_len, batched=True)
        neg_part = dataset.filter(filter_neg_label, batched=True)
        pos_part = dataset.filter(filter_pos_label, batched=True)
        save_path = os.path.join(split_path, domain)
        os.makedirs(save_path, exist_ok=True)
        dataset_dict = DatasetDict()
        dataset_dict["neg"] = neg_part
        dataset_dict["pos"] = pos_part
        dataset_dict.save_to_disk(save_path)


if __name__ == '__main__':
    main()
