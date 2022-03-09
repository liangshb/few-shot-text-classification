from gensim.models import word2vec, fasttext
import os
import pickle
import random
import torch
import numpy as np
from omegaconf import OmegaConf


def get_texts(train_loader, vocabulary):
    texts = []
    for filename in train_loader.loaders:
        for value in train_loader.loaders[filename]:
            loader = list(train_loader.loaders[filename][value])
            for data, _, _ in loader:
                for text in data:
                    text = text.tolist()
                    for i in range(len(text)):
                        text[i] = vocabulary.to_word(text[i])
                    texts.append(text)
    print('texts', len(texts))
    return texts


def get_weights(model, vocabulary, embed_dim):
    weights = np.zeros((len(vocabulary), embed_dim))
    for i in range(len(vocabulary)):
        if vocabulary.to_word(i) == '<pad>':
            continue
        if vocabulary.to_word(i) == '<unk>':
            continue
        weights[i] = model.wv[vocabulary.to_word(i)]
    return weights


def main():
    data_path = config['data']['path']
    weights_path = os.path.join(data_path, config["data"]["weights"])
    if not os.path.exists(weights_path):
        embed_dim = int(config['model']['encoder']['embed_dim'])
        vocabulary = pickle.load(open(os.path.join(data_path, config['data']['vocabulary']), 'rb'))
        train_loader = pickle.load(open(os.path.join(data_path, config['data']['train_loader']), 'rb'))
        texts = get_texts(train_loader, vocabulary)
        if config["data"]["pretrain"] == "word2vec":
            model = word2vec.Word2Vec(window=int(config['data']['window']), min_count=int(config['data']['min_count']),
                                      vector_size=embed_dim)
        else:
            model = fasttext.FastText(window=int(config['data']['window']), min_count=int(config['data']['min_count']),
                                      vector_size=embed_dim)
        model.build_vocab(texts)
        model.train(texts, total_examples=model.corpus_count, epochs=model.epochs)
        weights = get_weights(model, vocabulary, embed_dim)
        pickle.dump(weights, open(os.path.join(data_path, config['data']['weights']), 'wb'))
    else:
        print("Weights exists")


if __name__ == '__main__':
    config = OmegaConf.load("sysevr_config_lstm_attn.yaml")

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
