import sys, os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from laplace import Laplace
import numpy as np

import math

import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from asdfghjkl.operations import Bias, Scale

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--seed", type=int)
parser.add_argument("--root_data_dir", type=str)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--output_dir", type=str)

##########################################################################################
# Wide ResNet (for WRN16-4)
##########################################################################################
# Adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/models/wrn.py

class FixupBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(FixupBasicBlock, self).__init__()
        self.bias1 = Bias()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bias2 = Bias()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias4 = Bias()
        self.scale1 = Scale()
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bias1(x))
        else:
            out = self.relu1(self.bias1(x))
        if self.equalInOut:
            out = self.bias3(self.relu2(self.bias2(self.conv1(out))))
        else:
            out = self.bias3(self.relu2(self.bias2(self.conv1(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.bias4(self.scale1(self.conv2(out)))
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class FixupNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(FixupNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class FixupWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0):
        super(FixupWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = FixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bias1 = Bias()
        # 1st block
        self.block1 = FixupNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = FixupNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = FixupNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bias2 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(conv.weight,
                                mean=0,
                                std=np.sqrt(2. / k) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight,
                                    mean=0,
                                    std=np.sqrt(2. / k))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bias1(self.conv1(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(self.bias2(out))

def compute_metrics_from_probs(probs, targets, n_bins=15):
    conf = probs.max(-1)
    acc = probs.argmax(-1) == targets

    nll = -np.log(probs[np.arange(targets.size), targets]).mean()
    bins = np.linspace(0.1, 1, n_bins + 1)
    acc_bi = np.zeros(n_bins)
    conf_bi = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = np.logical_and(conf <= bins[i + 1], conf > bins[i])
        if np.sum(mask) > 0:
            bin_counts[i] = mask.sum()
            conf_bi[i] = conf[mask].mean()
            acc_bi[i] = acc[mask].mean()

    ECE = (np.abs(acc_bi - conf_bi) * bin_counts / np.sum(bin_counts)).sum()

    return acc.mean(), nll, ECE, bin_counts, acc_bi, conf_bi


@torch.no_grad()
def predict(dataloader, model):
    py = []

    for x, _ in dataloader:
        py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    seed = int(args.seed)

    model = FixupWideResNet(16, 4, 10, dropRate=0.3).cuda()

    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    batch_size = 1024

    trainset = torchvision.datasets.CIFAR10(root=args.root_data_dir, train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root=args.root_data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    _, val_set = torch.utils.data.random_split(
        testset, [9000, 1000], generator=torch.Generator().manual_seed(42)
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    val_targets = torch.cat([y for x, y in val_loader], dim=0).numpy()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    nlls = []
    checkpoint_paths = []

    for epoch in range(1000): #1000 epoch training: let's try to get a good MAP solution.
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

        if epoch % 25 == 24:
            model.eval()
            torch.save({"model_state_dict": model.state_dict()},
                       f"{args.checkpoint_dir}/wrn_16_4_fixup_epoch_{epoch}_seed_{seed}.pt")
            probs = predict(val_loader, model)
            _, nll, _, _, _, _ = compute_metrics_from_probs(
                probs, val_targets
            )

            nlls.append(nll)
            checkpoint_paths.append(f"{args.checkpoint_dir}/wrn_16_4_fixup_epoch_{epoch}_seed_{seed}.pt")
            model.train()

    best_checkpoint_path = checkpoint_paths[np.argmin(np.array(nlls))]
    print(f"Best checkpoint path: {best_checkpoint_path}")

    # reload best checkpoint path
    model.load_state_dict(
        torch.load(
            best_checkpoint_path,
            map_location="cuda:0",
        )["model_state_dict"]
    )

    torch.save({"model_state_dict": model.state_dict()},
               f"{args.checkpoint_dir}/best_seed{seed}.pt")