import argparse
from omegaconf import OmegaConf
import pickle
import os
import torch
from torch import optim
import random
import numpy as np
from model import FewShotInduction
from criterion import Criterion, Metrics

from tqdm import tqdm
from models.utils import get_encoder


def train_test():
    model.eval()
    loss_list = []
    for data, length, target in tqdm(test_loader):
        data = data.to(device)
        length = length.to(device)
        target = target.to(device)
        predict = model(data, length)
        loss, _ = criterion(predict, target)
        test_metrics.update(predict, target)
        loss_list.append(loss.item())
    results = test_metrics.get_metrics(reset=True)
    results["loss"] = torch.mean(torch.tensor(loss_list)).item()
    print('Test: Mcc: {} F1: {}'.format(results["mcc"], results["f1"]))
    return results["mcc"]


def main():
    state_dict = torch.load(config["model"]["model_path"])
    model.load_state_dict(state_dict)
    train_test()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("conf_path", type=str)
    arg_parser.add_argument("dev_path", type=str)
    args = arg_parser.parse_args()
    conf_path = args.conf_path
    dev_path = args.dev_path

    config = OmegaConf.load(conf_path)
    # seed
    seed = int(config['model']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # log_interval
    log_interval = int(config['model']['log_interval'])
    dev_interval = int(config['model']['dev_interval'])

    # data loaders
    train_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['train_loader']), 'rb'))
    dev_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'rb'))
    test_loader = pickle.load(open(dev_path, 'rb'))

    vocabulary = pickle.load(open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'rb'))

    # word2vec weights
    weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))

    # model & optimizer & criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support = int(config['model']['support'])
    encoder = get_encoder(config["model"], len(vocabulary), weights, vocabulary.padding_idx)
    model = FewShotInduction(C=int(config['model']['class']),
                             S=support,
                             encoder=encoder,
                             iterations=int(config['model']['iterations']),
                             outsize=int(config['model']['relation_dim']),
                             ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['model']['lr']))
    criterion = Criterion(way=int(config['model']['class']),
                          shot=int(config['model']['support']))
    dev_metrics = Metrics(way=int(config['model']['class']),
                          shot=int(config['model']['support']))
    test_metrics = Metrics(way=int(config['model']['class']),
                           shot=int(config['model']['support']))
    main()
