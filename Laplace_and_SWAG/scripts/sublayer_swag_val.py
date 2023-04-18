import sys, os

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from PIL import Image

import pickle
from copy import deepcopy
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from asdfghjkl.operations import Bias, Scale

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--stochastic_groups_str", type=str)
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
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bias2 = Bias()
        self.relu2 = nn.ReLU(inplace=True)
        self.bias3 = Bias()
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bias4 = Bias()
        self.scale1 = Scale()
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

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
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class FixupWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0):
        super(FixupWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = FixupBasicBlock
        # 1st conv before any network block
        self.num_layers = n * 3
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bias1 = Bias()
        # 1st block
        self.block1 = FixupNetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        # 2nd block
        self.block2 = FixupNetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        # 3rd block
        self.block3 = FixupNetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # global average pooling and classifier
        self.bias2 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                conv = m.conv1
                k = conv.weight.shape[0] * np.prod(conv.weight.shape[2:])
                nn.init.normal_(
                    conv.weight,
                    mean=0,
                    std=np.sqrt(2.0 / k) * self.num_layers ** (-0.5),
                )
                nn.init.constant_(m.conv2.weight, 0)
                if m.convShortcut is not None:
                    cs = m.convShortcut
                    k = cs.weight.shape[0] * np.prod(cs.weight.shape[2:])
                    nn.init.normal_(cs.weight, mean=0, std=np.sqrt(2.0 / k))
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


class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


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
def predict(dataloader, model, swag_results=None, swag=False, num_mc_swag=30):

    py = []
    device = next(model.parameters()).device

    if swag:
        theta_swa = swag_results["theta_swa"]
        sigma_diag = swag_results["sigma_diag"]
        D = swag_results["D"]
        K = swag_results["K"]

        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            all_probs_batch = torch.zeros((num_mc_swag, targets.numel(), 10)).to(device)

            for i in range(num_mc_swag):
                z1 = torch.normal(mean=torch.zeros((theta_swa.numel())), std=1.0).to(device)
                z2 = torch.normal(mean=torch.zeros(K), std=1.0).to(device)

                theta_i = theta_swa + 2 ** -0.5 * (sigma_diag ** 0.5 * z1) + (2 * (K - 1)) ** -0.5 * (
                            D @ z2[:, None]).flatten()
                torch.nn.utils.vector_to_parameters(theta_i, model.parameters())
                all_probs_batch[i, :, :] = torch.softmax(model(inputs), axis=-1)

            py.append(all_probs_batch.mean(axis=0))

    else:
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            py.append(torch.softmax(model(inputs), dim=-1))

    return torch.cat(py).cpu().numpy()


def generate_results_dict(all_dataloaders_dict, model, test_targets, swag_results=None, swag=False):
    res_dict = {}

    for name, dls in all_dataloaders_dict.items():
        print(f"Working on {name}")
        res_dict[name] = defaultdict(list)
        for dl in dls:
            probs = predict(dl, model, swag_results=swag_results, swag=swag)
            acc, nll, ece, counts, accs, confs = compute_metrics_from_probs(
                probs, test_targets
            )
            res_dict[name]["acc"].append(acc)
            res_dict[name]["nll"].append(nll)
            res_dict[name]["ece"].append(ece)
            res_dict[name]["counts"].append(counts)
            res_dict[name]["accs"].append(accs)
            res_dict[name]["confs"].append(confs)

    return res_dict


def run_swag(model, lr=1e-2, momentum=0.9, weight_decay=3e-4, K=20, n_epochs=10, snapshots_per_epoch=4):
    # hparams are default SWAG params (for imagenet), except slightly higher learning rate
    def _param_vector(model):
        return nn.utils.parameters_to_vector(model.parameters()).detach()

    batch_snapshot_freq = int(len(train_loader) / snapshots_per_epoch)

    # create a copy of the model to avoid undesired changes to the original model parameters
    _model = deepcopy(model)
    _model.eval()  # this model has no batchnorm, keep it with dropout meaned out
    # _model.train()
    device = next(_model.parameters()).device

    # initialize running estimates of first and second moment of model parameters
    # for convenience, just do it for all parameters. the optimiser will automatically
    # only update the ones that are needed
    mean = torch.zeros_like(_param_vector(_model))
    sq_mean = torch.zeros_like(_param_vector(_model))
    deviations = []

    n_snapshots = 0

    optimizer = torch.optim.SGD(
        [p for p in _model.parameters() if p.requires_grad], lr=lr, momentum=momentum, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    average_losses = []

    pbar = tqdm(range(n_epochs))
    for _ in pbar:
        def snapshot(n_snapshots_lambda, model_lambda, mean_lambda, sq_mean_lambda, deviations_lambda, K):
            old_fac, new_fac = n_snapshots_lambda / (n_snapshots_lambda + 1), 1 / (n_snapshots_lambda + 1)
            mean_lambda = mean_lambda * old_fac + _param_vector(model_lambda) * new_fac
            sq_mean_lambda = sq_mean_lambda * old_fac + _param_vector(model_lambda) ** 2 * new_fac
            deviation = _param_vector(model_lambda) - mean_lambda

            if len(deviations_lambda) == K:
                deviations_lambda.pop(0)
            deviations_lambda.append(deviation)

            return n_snapshots_lambda + 1, mean_lambda, sq_mean_lambda, deviations_lambda

        epoch_losses = []
        for batch_i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(_model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.cpu().clone().detach().numpy())

            if batch_i % batch_snapshot_freq == (batch_snapshot_freq - 1):
                n_snapshots, mean, sq_mean, deviations = snapshot(n_snapshots, _model, mean, sq_mean, deviations, K)
                pbar.set_postfix({'Running epoch_loss': np.mean(epoch_losses), 'snapshots': n_snapshots})

        average_losses.append(np.mean(epoch_losses))

    D = torch.zeros((_param_vector(_model).numel(), K))
    for i in range(K):
        D[:, i] = deviations[i]

    return {
        "theta_swa": mean.to(device),
        "sigma_diag": (sq_mean - mean ** 2).to(device).clamp(1e-14),
        "D": D.to(device),
        "K": K
    }

if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    batch_size = 1024 # can now be large batchsize

    trainset = torchvision.datasets.CIFAR10(
        root=args.root_data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.root_data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

    _, val_set = torch.utils.data.random_split(
        testset, [9000, 1000], generator=torch.Generator().manual_seed(42)
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    val_targets = torch.cat([y for x, y in val_loader], dim=0).numpy()

    corruptions = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
        "glass_blur",
    ]
    corrupted_dataloaders_dict = {}

    np_y = np.load(f"{args.root_data_dir}/CIFAR-10-C/labels.npy").astype(
        np.int64
    )
    for c in corruptions:
        data_file = f"{args.root_data_dir}/CIFAR-10-C/{c}.npy"
        np_x = np.load(data_file)

        dataset = DatafeedImage(np_x, np_y, test_transform)
        dls = []
        for i in [1, 2, 3, 4, 5]:
            subset_dataset = torch.utils.data.Subset(
                dataset, np.arange(10000 * (i - 1), 10000 * i)
            )
            dl = torch.utils.data.DataLoader(
                subset_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            dls.append(dl)

        corrupted_dataloaders_dict[c] = dls

    all_dataloaders_dict = {"no_corruption": [test_loader]}
    all_dataloaders_dict = {**all_dataloaders_dict, **corrupted_dataloaders_dict}

    def reset_model():
        model.load_state_dict(
            torch.load(
                f"{args.checkpoint_dir}/best_seed{args.seed}.pt",
                map_location="cuda:0",
            )["model_state_dict"]
        )
        model.eval()

        groups = [
            [0, 1],
            list(range(2, 10)),
            list(range(10, 17)),
            list(range(17, 25)),
            list(range(25, 32)),
            list(range(32, 40)),
            list(range(40, 48)),
            [48, 49],
        ]

        layer_indices = []

        for i, c in enumerate(args.stochastic_groups_str):
            if c == "1":
                layer_indices.extend(groups[i])

        for layer_i, (name, params) in enumerate(model.named_parameters()):
            if layer_i in layer_indices:
                params.requires_grad = True
            else:
                params.requires_grad = False


    model = FixupWideResNet(16, 4, 10, dropRate=0.3).cuda()

    all_res_dict = {
        "seed": args.seed,
    }

    if args.stochastic_groups_str == "00000000":
        reset_model()
        all_res_dict["test_results"] = generate_results_dict(
            all_dataloaders_dict, model, test_targets=test_targets, swag=False
        )
    else:
        # sweep learning rate
        lr_sweep = np.logspace(-5, -2, num=25, endpoint=True) # bigger sweep of values here, now.
        holdout_likelihood_sweep = np.zeros_like(lr_sweep)
        all_swag_results = []

        for i in range(25):
            reset_model()
            swag_results = run_swag(model, lr=lr_sweep[i])
            all_swag_results.append(swag_results)
            probs = predict(val_loader, model, swag_results=swag_results, swag=True, num_mc_swag=30)
            _, nll, _, _, _, _ = compute_metrics_from_probs(
                probs, val_targets
            )
            holdout_likelihood_sweep[i] = -nll

        all_res_dict["best_lr"] = float(lr_sweep[np.nanargmax(holdout_likelihood_sweep)])
        best_swag_results = all_swag_results[np.nanargmax(holdout_likelihood_sweep)]

        all_res_dict["test_results"] = generate_results_dict(
            all_dataloaders_dict, model, test_targets=test_targets, swag=True, swag_results=best_swag_results
        )

    fname = f"{args.output_dir}/sublayer_swag/{args.stochastic_groups_str}_s{args.seed}_val"
    pickle.dump(all_res_dict, open(f"{fname}.pkl", "wb"))
