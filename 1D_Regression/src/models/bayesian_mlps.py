import numpy as np
import torch.distributions
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import src.utils as utils
from src.layers.intel_bayesian_layers import LinearFlipout, BaseVariationalLayer_

log = utils.get_logger(__name__)


def compute_kl_distributions(pl_module):
    kl_dict = {}
    for k, v in pl_module.named_modules():
        if any(
            isinstance(v, bayesian_layer)
            for bayesian_layer in [
                LinearFlipout,
            ]
        ):
            with torch.inference_mode():
                kl_w, kl_b = v.compute_weight_bias_kls()
            kl_dict[f"kl_{k}_w"] = kl_w.detach()
            kl_dict[f"kl_{k}_b"] = kl_b.detach()

    if len(kl_dict) > 0:
        kl_agg = torch.cat(list(kl_dict.values()))
        kl_dict["kl_agg"] = kl_agg

    return kl_dict

def get_nonlinearity_from_string(non_linearity):
    if non_linearity == "leaky_relu":
        return F.leaky_relu
    elif non_linearity == "tanh":
        return F.tanh
    elif non_linearity == "relu":
        return F.relu
    elif non_linearity == "silu":
        return F.silu
    elif non_linearity == "elu":
        return F.elu

class BayesianMLP(pl.LightningModule):
    def __init__(
        self,
        N_train_dataset,
        width=100,
        n_mc_train=10,  # 10 MC samples is commonly used for linear layers
        n_mc_eval=100,
        output_noise_scale=1.0,
        layer_class=LinearFlipout,
        layer_kwargs=None,
        input_dim=1,
        output_dim=1,
        non_linearity="leaky_relu",
        beta=1.0,
        lr=1e-3,
        weight_decay=0,
        beta_warmup=False,
    ):
        super().__init__()

        if not layer_kwargs:
            layer_kwargs = {}

        self.fc1 = layer_class(input_dim, width, **layer_kwargs)
        self.fc2 = layer_class(width, width, **layer_kwargs)
        self.fc3 = layer_class(width, width, **layer_kwargs)
        self.output_layer = layer_class(width, output_dim, **layer_kwargs)

        self.save_hyperparameters()

        self.loss = nn.GaussianNLLLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

        if not beta_warmup:
            self.beta_lambda = lambda epochs: beta
        else:
            self.beta_lambda = lambda epochs: beta * np.clip(
                (epochs + 1) / 4000.0, a_min=1e-3, a_max=1.0
            )

    def forward(self, x, mode="sample"):
        kl = 0
        x, kl_i = self.fc1(x, mode=mode)
        x = self.non_linearity(x)
        kl += kl_i

        x, kl_i = self.fc2(x, mode=mode)
        x = self.non_linearity(x)
        kl += kl_i

        x, kl_i = self.fc3(x, mode=mode)
        x = self.non_linearity(x)
        kl += kl_i

        y, kl_i = self.output_layer(x, mode=mode)
        kl += kl_i

        return y, kl

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        pred_losses = torch.zeros(self.hparams.n_mc_train)
        kls = torch.zeros(self.hparams.n_mc_train)
        mses = torch.zeros(self.hparams.n_mc_train)

        output_var = torch.ones_like(y) * (self.hparams.output_noise_scale ** 2)

        for i in range(self.hparams.n_mc_train):
            pred, kl = self.forward(x, mode="sample")
            pred_loss = self.loss(pred, y, output_var)
            mses[i] = self.mse(pred, y)
            pred_losses[i] = pred_loss
            kls[i] = kl

        mean_pred_loss = torch.mean(pred_losses)
        mean_kl = torch.mean(kls)
        mean_mse = torch.mean(mses)

        beta = self.beta_lambda(self.current_epoch)

        negative_beta_elbo = (
            mean_pred_loss + (beta / self.hparams.N_train_dataset) * mean_kl
        )
        expected_log_prob = -mean_pred_loss
        kl = (1 / self.hparams.N_train_dataset) * mean_kl
        beta_kl = (beta / self.hparams.N_train_dataset) * mean_kl

        self.log(
            "train/negative_beta_elbo",
            negative_beta_elbo,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train/expected_log_prob",
            expected_log_prob,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/beta",
            beta,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/expected_mse",
            mean_mse,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/kl",
            kl,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/beta_kl",
            beta_kl,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train/abs_beta_kl_div_abs_expect_log_prob",
            torch.abs(beta_kl) / torch.abs(expected_log_prob),
            on_step=False,
            on_epoch=True,
        )

        return negative_beta_elbo

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        pred_losses = torch.zeros(self.hparams.n_mc_eval)
        mses = torch.zeros(self.hparams.n_mc_eval)
        output_var = torch.ones_like(y) * (self.hparams.output_noise_scale ** 2)

        for i in range(self.hparams.n_mc_eval):
            pred, _ = self.forward(x)
            pred_loss = self.loss(pred, y, output_var)
            mses[i] = self.mse(pred, y)
            pred_losses[i] = pred_loss

        mean_expected_log_prob = -torch.mean(pred_losses)
        mean_mse = torch.mean(mses)

        self.log(
            "val/expected_log_prob",
            mean_expected_log_prob,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val/expected_mse",
            mean_mse,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        pred_losses = torch.zeros(self.hparams.n_mc_eval)
        output_var = torch.ones_like(y) * (self.hparams.output_noise_scale ** 2)

        for i in range(self.hparams.n_mc_eval):
            pred, _ = self.forward(x)
            pred_loss = self.loss(pred, y, output_var)
            pred_losses[i] = pred_loss

        mean_expected_log_prob = -torch.mean(pred_losses)

        self.log(
            "test/expected_log_prob",
            mean_expected_log_prob,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.hparams.weight_decay
        )

        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @property
    def model_name(self):
        return "BayesianMLP"

    @property
    def n_layers_stochastic(self):
        return 4

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=("mu", "rho")
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                log.info(f"{name} excluded from weight decay")
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]


class MixedStochasticMLP(BayesianMLP):
    def __init__(
        self,
        N_train_dataset,
        width=100,
        n_mc_train=10,  # 10 MC samples is commonly used for linear layers
        n_mc_eval=100,
        output_noise_scale=1.0,
        stochastic_layer_code=15,
        stochastic_layer_class=LinearFlipout,
        stochastic_layer_kwargs=None,
        deterministic_layer_class=nn.Linear,
        deterministic_layer_kwargs=None,
        input_dim=1,
        output_dim=1,
        non_linearity="leaky_relu",
        beta=1.0,
        weight_decay=1e-4,
        beta_warmup=False,
        lr=1e-3,
        *args,
        **kwargs,
    ):
        super(
            BayesianMLP, self
        ).__init__()  # we cannot call the parent init, because it will instantiate layers that we do not want. Call grandparent init. but we still want to subclass, because many of the methods here are shared!

        if not deterministic_layer_kwargs:
            deterministic_layer_kwargs = {}

        if not stochastic_layer_kwargs:
            stochastic_layer_kwargs = {}

        assert stochastic_layer_code in list(range(16))
        self.stochastic_layer_flags = [
            stochastic_layer_code & 2 ** i > 0 for i in range(0, 4)
        ]
        log.info(
            f"Stochastic layer code {stochastic_layer_code} converted into {self.model_name}"
        )

        self.fc1 = (
            stochastic_layer_class(input_dim, width, **stochastic_layer_kwargs)
            if self.stochastic_layer_flags[0]
            else deterministic_layer_class(
                input_dim, width, **deterministic_layer_kwargs
            )
        )
        self.fc2 = (
            stochastic_layer_class(width, width, **stochastic_layer_kwargs)
            if self.stochastic_layer_flags[1]
            else deterministic_layer_class(width, width, **deterministic_layer_kwargs)
        )

        self.fc3 = (
            stochastic_layer_class(width, width, **stochastic_layer_kwargs)
            if self.stochastic_layer_flags[2]
            else deterministic_layer_class(width, width, **deterministic_layer_kwargs)
        )

        self.output_layer = (
            stochastic_layer_class(width, output_dim, **stochastic_layer_kwargs)
            if self.stochastic_layer_flags[3]
            else deterministic_layer_class(
                width, output_dim, **deterministic_layer_kwargs
            )
        )
        self.non_linearity = get_nonlinearity_from_string(non_linearity)

        self.loss = nn.GaussianNLLLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

        if not beta_warmup:
            self.beta_lambda = lambda epochs: beta
        else:
            self.beta_lambda = lambda epochs: beta * np.clip(
                (epochs + 1) / 4000.0, a_min=1e-3, a_max=1.0
            )

        self.save_hyperparameters()

    def forward(self, x, mode="sample"):
        kl = 0

        def custom_layer_forward(x, layer):
            if isinstance(layer, BaseVariationalLayer_):
                return layer(x, mode)
            else:
                return layer(x), 0  # return zero regularisation loss

        x, kl_i = custom_layer_forward(x, self.fc1)
        x = self.non_linearity(x)
        kl += kl_i

        x, kl_i = custom_layer_forward(x, self.fc2)
        x = self.non_linearity(x)
        kl += kl_i

        x, kl_i = custom_layer_forward(x, self.fc3)
        x = self.non_linearity(x)
        kl += kl_i

        y, kl_i = custom_layer_forward(x, self.output_layer)
        kl += kl_i

        return y, kl

    @property
    def model_name(self):
        name_str = ",".join(
            ["Stoc" if f else "Det" for f in self.stochastic_layer_flags]
        )
        name_str = f"I:{name_str}:O MixedStochasticMLP"
        return name_str

    @property
    def n_layers_stochastic(self):
        return sum(self.stochastic_layer_flags)