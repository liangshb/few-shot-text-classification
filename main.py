from omegaconf import OmegaConf
import pickle
import os
import torch
from torch import optim
import random
import numpy as np
from model import FewShotInduction
from criterion import Criterion, Metrics
from tensorboardX import SummaryWriter

from tqdm import tqdm
from models.utils import get_encoder


def train(episode):
    model.train()
    data, length, target = train_loader.get_batch()
    data = data.to(device)
    length = length.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    predict = model(data, length)
    loss, acc = criterion(predict, target)
    loss.backward()
    optimizer.step()

    writer.add_scalar('train_loss', loss.item(), episode)
    writer.add_scalar('train_acc', acc, episode)
    if episode % log_interval == 0:
        print('Train Episode: {} Loss: {} Acc: {}'.format(episode, loss.item(), acc))


def log_metrics(results, prefix, episode=None):
    for key, value in results.items():
        writer.add_scalar(f"{prefix}_{key}", value, episode)


def dev(episode):
    model.eval()
    loss_list = []
    for data, length, target in tqdm(dev_loader):
        data = data.to(device)
        length = length.to(device)
        target = target.to(device)
        predict = model(data, length)
        loss, _ = criterion(predict, target)
        dev_metrics.update(predict, target)
        loss_list.append(loss.item())
    results = dev_metrics.get_metrics(reset=True)
    results["loss"] = torch.mean(torch.tensor(loss_list))
    log_metrics(results, "dev", episode)
    print('Dev Episode: {} Mcc: {} F1: {}'.format(episode, results["mcc"], results["f1"]))
    return results["mcc"]


def test():
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
    results["loss"] = torch.mean(torch.tensor(loss_list))
    log_metrics(results, "test")
    print('Test: Mcc: {} F1: {}'.format(results["mcc"], results["f1"]))
    return results["mcc"]


def main():
    best_episode, best_acc = 0, 0.
    episodes = int(config['model']['episodes'])
    early_stop = int(config['model']['early_stop']) * dev_interval
    for episode in range(1, episodes + 1):
        train(episode)
        if episode % dev_interval == 0:
            acc = dev(episode)
            if acc > best_acc:
                print('Better acc! Saving model!')
                torch.save(model.state_dict(), config['model']['model_path'])
                best_episode, best_acc = episode, acc
            if episode - best_episode >= early_stop:
                print('Early stop at episode', episode)
                break

    print('Reload the best model on episode', best_episode, 'with best acc', best_acc.item())
    ckpt = torch.load(config['model']['model_path'])
    model.load_state_dict(ckpt)
    test()


if __name__ == "__main__":
    # config
    config = OmegaConf.load("sysevr_config_lstm_attn.yaml")

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
    test_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['test_loader']), 'rb'))

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

    # writer
    os.makedirs(config['model']['log_path'], exist_ok=True)
    writer = SummaryWriter(config['model']['log_path'])
    main()
    writer.close()
