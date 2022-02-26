from torch.utils.data import Dataset


class LenDataset(Dataset):
    def __init__(
            self,
            data_dict,
    ):
        super().__init__()
        self.data = data_dict["data"]
        self.length = data_dict["length"]
        self.target = data_dict["target"]
        self.len = len(self.target)

    def __getitem__(self, index):
        return self.data[index], self.length[index], self.target[index]

    def __len__(self):
        return self.len
