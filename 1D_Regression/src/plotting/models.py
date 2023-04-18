import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"pdf.fonttype": 42, "figure.figsize": (4, 3), "figure.dpi": 200})

def plot_kl_dict(kl_dict, title=None):
    plt.figure(figsize=(8, 8), dpi=300)

    nonempty_dicts = [(k, v) for k,v in kl_dict.items() if v.numel() > 0 ]
    n_grid = int(len(nonempty_dicts) ** 0.5) + 1

    for sp_i, (k, v) in enumerate(list(nonempty_dicts)):
        plt.subplot(n_grid, n_grid, sp_i + 1)
        kls = v.detach().cpu().numpy()
        sns.histplot(kls, bins=60)
        plt.yscale('log')
        plt.xlim(-0.1, np.max(kls) * 1.1)
        plt.xlabel("$D_{KL}$")
        plt.ylabel("count")
        plt.title(k)

    if title is not None:
        plt.suptitle(title, wrap=True)

    plt.tight_layout()
    return plt
