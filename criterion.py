import torch
from torch.nn.modules.loss import _Loss
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, Specificity


class Criterion(_Loss):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target):  # (Q,C) (Q)
        target = target[self.amount:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc


class Metrics:
    def __init__(self, way, shot):
        self.amount = way * shot
        self.acc = Accuracy()
        self.precision = Precision(average=None, num_classes=2)
        self.recall = Recall(average=None, num_classes=2)
        self.f1 = F1Score(average=None, num_classes=2)
        self.mcc = MatthewsCorrCoef(num_classes=2)
        self.spec = Specificity(average=None, num_classes=2)

    def update(self, probs, target):
        target = target[self.amount:].to("cpu")
        preds = torch.argmax(probs, dim=1).to("cpu")
        self.acc.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)
        self.mcc.update(preds, target)
        self.spec.update(preds, target)

    def get_metrics(self, reset: bool = False):
        results = {
            "accuracy": self.acc.compute(),
            "precision": self.precision.compute()[1],
            "recall": self.recall.compute()[1],
            "f1": self.f1.compute()[1],
            "mcc": self.mcc.compute(),
            "spec": self.spec.compute()[1],
        }
        if reset:
            self.acc.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1.reset()
            self.mcc.reset()
            self.spec.reset()
        return results
