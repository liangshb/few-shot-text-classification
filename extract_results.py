import os
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def main():
    ea = event_accumulator.EventAccumulator(events_path)
    ea.Reload()
    dev_accuracy = [item.value for item in ea.scalars.Items("dev_accuracy")]
    dev_precision = [item.value for item in ea.scalars.Items("dev_precision")]
    dev_recall = [item.value for item in ea.scalars.Items("dev_recall")]
    dev_f1 = [item.value for item in ea.scalars.Items("dev_f1")]
    dev_mcc = [item.value for item in ea.scalars.Items("dev_mcc")]
    dev_spec = [item.value for item in ea.scalars.Items("dev_spec")]

    index = np.argmax(np.array(dev_mcc))
    accuracy = dev_accuracy[index]
    precision = dev_precision[index]
    recall = dev_recall[index]
    f1 = dev_f1[index]
    mcc = dev_mcc[index]
    spec = dev_spec[index]
    print(f"fpr: {1 - spec}")
    print(f"fnr: {1 - recall}")
    print(f"acc: {accuracy}")
    print(f"pre: {precision}")
    print(f"f1: {f1}")
    print(f"mcc: {mcc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    path = args.path

    files = os.listdir(path)
    events_file = [file for file in files if "events" in file][0]
    events_path = os.path.join(path, events_file)
    main()
