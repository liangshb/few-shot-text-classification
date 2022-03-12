import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_ea(ea_path):
    ea = event_accumulator.EventAccumulator(ea_path)
    ea.Reload()
    train_results = {}
    dev_results = {}
    for key in [key for key in ea.scalars.Keys() if "train" in key]:
        train_results[key] = [item.value for item in ea.scalars.Items(key)]
    for key in [key for key in ea.scalars.Keys() if "train" in key]:
        dev_results[key] = [item.value for item in ea.scalars.Items(key)]
    return {
        "train": train_results,
        "dev": dev_results
    }


def main():
    train_writer = pd.ExcelWriter(os.path.join(path, f"{pretrain}_train.xlsx"))
    dev_writer = pd.ExcelWriter(os.path.join(path, f"{pretrain}_dev.xlsx"))
    models_path = [file for file in os.listdir(path) if "xlsx" not in file]
    for model in models_path:
        pretrain_path = os.path.join(path, model, pretrain)
        ea_file = [file for file in os.listdir(pretrain_path) if "events" in file][-1]
        ea_path = os.path.join(pretrain_path, ea_file)
        results = extract_ea(ea_path)
        train_df = pd.DataFrame(results["train"])
        dev_df = pd.DataFrame(results["dev"])
        train_df.to_excel(train_writer, sheet_name=model)
        dev_df.to_excel(dev_writer, sheet_name=model)
    train_writer.save()
    dev_writer.save()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", type=str)
    arg_parser.add_argument("pretrain", type=str)
    args = arg_parser.parse_args()
    path = args.path
    pretrain = args.pretrain

    main()
