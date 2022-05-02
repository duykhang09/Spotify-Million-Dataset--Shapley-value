from random import shuffle
from numpy import dtype
import torch
from torch.utils.data import Dataset
import pandas as pd

top_song_count = 2000
pl_top_trk_df = pd.read_csv('data.csv', names=['pid', 'followers'] + [i for i in range(0, top_song_count)], header=0)
test = pl_top_trk_df.sample(frac=0.1, axis=0)
train = pl_top_trk_df.drop(index=test.index)

class TopSongsTrain(Dataset):
    def __init__(self):
        super(TopSongsTrain)

        x=train[[i for i in range(0, top_song_count)]].values
        y=train[['followers']].values

        self.x_train=torch.tensor(x, dtype=torch.float32)
        self.y_train=torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class TopSongsTest(Dataset):
    def __init__(self):
        x=test[[i for i in range(0, top_song_count)]].values
        y=test[['followers']].values

        self.x_train=torch.tensor(x, dtype=torch.float32)
        self.y_train=torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        ## Remove a few hots to cross validate...
        return self.x_train[idx], self.y_train[idx]
