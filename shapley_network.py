from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

top_song_count = 2000

import playlist_dataset

class VNet(nn.Module):

    def __init__(self):
        super(VNet, self).__init__()

        self.input_layer = nn.Linear(top_song_count, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # take in 2000 song input to input layer
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main():
    net = VNet()
    net.to(device)

    # Hyper-parameters
    learning_rate = 0.02
    batch_size = 128
    epochs = 100
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------")
        train_loop(train_dataloader, net, loss_fn, optimizer, cur_epoch=t)
        test_loop(test_dataloader, net, loss_fn, cur_epoch=t)
    print("Done!")
    torch.save(net, 'model-relu-64-16.pth')


def train_loop(dataloader, model , loss_fn, optimizer, cur_epoch):
    running_loss = 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred_followers = model(X.to(device))
        loss = loss_fn(pred_followers, y.to(device))

        # Backprop by grad = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 99:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>6f} [{current:>5d}/{size:>5d}]")

        writer.add_scalar("Train Loss", running_loss / 100, cur_epoch * len(dataloader) + batch)
        running_loss = 0.0

def test_loop(dataloader, model, loss_fn, cur_epoch):
    size = len(dataloader.dataset)
    test_loss = 0

    # No grad backprop optimiz update
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred_followers = model(X.to(device))
            test_loss += loss_fn(pred_followers, y.to(device)).item()

    test_loss /= size

    print(f"Avg loss: {test_loss:>6f}")
    writer.add_scalar('Test loss', test_loss, cur_epoch * len(dataloader) + batch)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('runs/model-relu-64-16')
    dataset = playlist_dataset.TopSongsTrain()
    test_dataset = playlist_dataset.TopSongsTest()
    main()