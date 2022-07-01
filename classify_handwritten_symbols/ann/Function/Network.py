import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary


# create ResNet
class ResNet(nn.Module):
    def __init__(self, channels):
        super(ResNet, self).__init__()
        self.channnels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + x


class model(nn.Module):
    def __init__(self, channel, n_blocks=1):
        super(model, self).__init__()
        # channel=[16,32,64,128,256]
        self.channel = channel  # a list

        self.block_across1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channel[0], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channel[0], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[0])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn1 = nn.BatchNorm2d(channel[0])

        self.block_across2 = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[1])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn2 = nn.BatchNorm2d(channel[1])

        self.block_across3 = nn.Sequential(
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[2])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn3 = nn.BatchNorm2d(channel[2])

        self.block_across4 = nn.Sequential(
            nn.Conv2d(in_channels=channel[2], out_channels=channel[3], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=channel[2], out_channels=channel[3], kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[3])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn4 = nn.BatchNorm2d(channel[3])

        self.block_across5 = nn.Sequential(
            nn.Conv2d(in_channels=channel[3], out_channels=channel[4], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=channel[3], out_channels=channel[4], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[4])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn5 = nn.BatchNorm2d(channel[4])

        self.block_across6 = nn.Sequential(
            nn.Conv2d(in_channels=channel[4], out_channels=channel[5], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=channel[4], out_channels=channel[5], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[5])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn6 = nn.BatchNorm2d(channel[5])

        self.block_across7 = nn.Sequential(
            nn.Conv2d(in_channels=channel[5], out_channels=channel[6], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels=channel[5], out_channels=channel[6], kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            *(n_blocks * [ResNet(channel[6])]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.bn7 = nn.BatchNorm2d(channel[6])
        #
        # self.flatten = nn.Flatten()
        #
        # self.classifier1 = nn.Sequential(
        #     nn.Linear(channel[3], channel[3]//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        #
        # self.classifier2 = nn.Sequential(
        #     nn.Linear(channel[3] // 2, channel[3]//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        #
        # self.classifier3 = nn.Sequential(
        #     nn.Linear(channel[3]//2, 25),
        #     nn.Softmax(dim=1)
        # )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=channel[6], out_channels=channel[6] // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[6] // 2, out_channels=channel[6] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel[6] // 4, out_channels=25, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.block1(x) + self.block_across1(x)
        out = self.bn1(out)
        out = self.block2(out) + self.block_across2(out)
        out = self.bn2(out)
        out = self.block3(out) + self.block_across3(out)
        out = self.bn3(out)
        out = self.block4(out) + self.block_across4(out)
        out = self.bn4(out)
        out = self.block5(out) + self.block_across5(out)
        out = self.bn5(out)
        out = self.block6(out) + self.block_across6(out)
        out = self.bn6(out)
        out = self.block7(out) + self.block_across7(out)
        out = self.bn7(out)

        # out = self.block6(out)
        # out = self.block7(out)
        # out = self.flatten(out)
        # out = self.classifier1(out)
        # out = self.classifier2(out)
        # out = self.classifier3(out)
        out = self.classifier(out)
        out = out.reshape(batch_size, -1)
        out = torch.softmax(out, dim=1)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channel = [8, 16, 32, 64, 128, 256, 512]
    model = model(channel).to(device)
    summary(model, (1, 150, 150))
