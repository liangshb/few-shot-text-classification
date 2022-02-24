import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.highway import Highway


class Encoder(nn.Module):
    def __init__(self, num_classes, num_support_per_class,
                 vocab_size, embed_size, hidden_size,
                 output_dim, weights):
        super(Encoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_size, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def attention(self, x):
        weights = torch.tanh(self.fc1(x))
        weights = self.fc2(weights)  # (batch=k*c, seq_len, d_a)
        batch, seq_len, d_a = weights.shape
        weights = weights.transpose(1, 2)  # (batch=k*c, d_a, seq_len)
        weights = weights.contiguous().view(-1, seq_len)
        weights = F.softmax(weights, dim=1).view(batch, d_a, seq_len)
        sentence_embeddings = torch.bmm(weights, x)  # (batch=k*c, d_a, 2*hidden)
        avg_sentence_embeddings = torch.mean(sentence_embeddings, dim=1)  # (batch, 2*hidden)
        return avg_sentence_embeddings

    def get_output_dim(self):
        return self.hidden_size * 2

    def forward(self, x, hidden=None):
        batch_size, _ = x.shape
        if hidden is None:
            h = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
            c = x.data.new(2, batch_size, self.hidden_size).fill_(0).float()
        else:
            h, c = hidden
        x = self.embedding(x)
        outputs, _ = self.bilstm(x, (h, c))  # (batch=k*c,seq_len,2*hidden)
        outputs = self.attention(outputs)  # (batch=k*c, 2*hidden)
        # (c*s, 2*hidden_size), (c*q, 2*hidden_size)
        support, query = outputs[0: self.num_support], outputs[self.num_support:]
        return support, query


class Induction(nn.Module):
    def __init__(self, C, S, H, iterations):
        super(Induction, self).__init__()
        self.C = C
        self.S = S
        self.H = H
        self.iterations = iterations
        self.W = torch.nn.Parameter(torch.randn(H, H))

    def forward(self, x):
        b_ij = torch.zeros(self.C, self.S).to(x)
        for _ in range(self.iterations):
            d_i = F.softmax(b_ij.unsqueeze(2), dim=1)  # (C,S,1)
            e_ij = torch.mm(x.reshape(-1, self.H), self.W).reshape(self.C, self.S, self.H)  # (C,S,H)
            c_i = torch.sum(d_i * e_ij, dim=1)  # (C,H)
            # squash
            squared = torch.sum(c_i ** 2, dim=1).reshape(self.C, -1)
            coeff = squared / (1 + squared) / torch.sqrt(squared + 1e-9)
            c_i = coeff * c_i
            c_produce_e = torch.bmm(e_ij, c_i.unsqueeze(2))  # (C,S,1)
            b_ij = b_ij + c_produce_e.squeeze(2)

        return c_i


class Relation(nn.Module):
    def __init__(self, C, H, out_size):
        super(Relation, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = torch.nn.Parameter(torch.randn(C * out_size, C))
        self.b = torch.nn.Parameter(torch.randn(C))

    def forward(self, class_vector, query_encoder):  # (C,H) (Q,H)
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,Q)
        V = F.relu(mid_pro.transpose(0, 1))  # (Q,C*out_size)
        probs = torch.sigmoid(torch.mm(V, self.W) + self.b)  # (Q,C)
        return probs


class FewShotInduction(nn.Module):
    def __init__(self, C, S, encoder,
                 iterations, outsize):
        super(FewShotInduction, self).__init__()
        self.encoder = encoder
        encoder_output_dim = encoder.get_output_dim()
        self.induction = Induction(C, S, encoder_output_dim, iterations)
        self.relation = Relation(C, encoder_output_dim, outsize)

    def forward(self, x):
        support_encoder, query_encoder = self.encoder(x)  # (k*c, 2*hidden_size)
        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs


class CNNEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 num_support_per_class,
                 vocab_size,
                 embed_size,
                 num_filters,
                 kernel_sizes,
                 dropout,
                 hidden_size,
                 weights
                 ):
        super(CNNEncoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0
        )
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_size,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            len(kernel_sizes) * num_filters, hidden_size
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        logits = self.norm(self.fc(x))
        support, query = logits[0: self.num_support], logits[self.num_support:]
        return support, query

    def get_output_dim(self):
        return self.hidden_size


class CNNHEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 num_support_per_class,
                 vocab_size,
                 embed_size,
                 num_filters,
                 kernel_sizes,
                 num_layers,
                 dropout,
                 hidden_size,
                 weights,
                 ):
        super(CNNHEncoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0
        )
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_size,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.highway = Highway(num_filters * len(kernel_sizes), num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            len(kernel_sizes) * num_filters, hidden_size
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.highway(x)
        logits = self.norm(self.fc(x))
        support, query = logits[0: self.num_support], logits[self.num_support:]
        return support, query

    def get_output_dim(self):
        return self.hidden_size


class LSTMEncoder(nn.Module):
    def __init__(self, num_classes, num_support_per_class,
                 vocab_size, embed_size, hidden_size, num_layers,
                 weights):
        super(LSTMEncoder, self).__init__()
        self.num_support = num_classes * num_support_per_class
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.bilstm(x)
        outputs = torch.mean(outputs, dim=1)
        support, query = outputs[0: self.num_support], outputs[self.num_support:]
        return support, query

    def get_output_dim(self):
        return self.hidden_size * 2


class GRUEncoder(nn.Module):
    def __init__(self, num_classes, num_support_per_class,
                 vocab_size, embed_size, hidden_size, num_layers,
                 weights):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_support = num_classes * num_support_per_class

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.bigru = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, _ = self.bigru(x)
        outputs = torch.mean(outputs, dim=1)
        support, query = outputs[0: self.num_support], outputs[self.num_support:]
        return support, query

    def get_output_dim(self):
        return self.hidden_size * 2
