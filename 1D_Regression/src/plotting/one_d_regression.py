import torch
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "figure.figsize": (4, 3), "figure.dpi": 200})


def plot_mean(network, datamodule, n_mc=50, n_grid=250, title=None, xlim=None, ylim=None, new_fig=True):

    with torch.inference_mode():
        if xlim is not None:
            x_test = (
                torch.linspace(xlim[0], xlim[1], n_grid).reshape((n_grid, 1)).to(device=network.device)
            )
        else:
            x_test = (
                torch.linspace(-3, 3, n_grid).reshape((n_grid, 1)).to(device=network.device)
            )
        mu_all = torch.zeros((n_grid, n_mc)).to(device=network.device)

        for i in range(n_mc):
            mu_i, _ = network.forward(x_test, mode="sample")
            mu_all[:, i] = mu_i.flatten()

    x_test = x_test.detach().cpu().numpy()
    mu_all = mu_all.detach().cpu().numpy()

    if new_fig:
        plt.figure()  # create new figure

    plt.plot(x_test, mu_all.mean(axis=-1), color="tab:blue")
    plt.fill_between(
        x_test.flatten(),
        mu_all.mean(axis=-1) - 2 * mu_all.std(axis=-1),
        mu_all.mean(axis=-1) + 2 * mu_all.std(axis=-1),
        alpha=0.1,
        color="tab:blue",
    )

    plt.scatter(
        datamodule.train_dataloader().dataset.X,
        datamodule.train_dataloader().dataset.Y,
        s=2,
        color="tab:orange",
    )
    plt.scatter(
        datamodule.val_dataloader().dataset.X,
        datamodule.val_dataloader().dataset.Y,
        s=3,
        color="tab:purple",
        marker="*",
    )
    plt.scatter(
        datamodule.test_dataloader().dataset.X,
        datamodule.test_dataloader().dataset.Y,
        s=3,
        color="tab:purple",
        marker="*",
    )

    if title is not None:
        plt.title(title, wrap=True)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()

    return plt
