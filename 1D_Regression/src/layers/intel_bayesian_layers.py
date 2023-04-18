# copied from Intel people
# https://github.com/IntelLabs/bayesian-torch
# under BSD-3-Clause license

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BaseVariationalLayer_(nn.Module):
    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, reduction="sum"):
        """
        Calculates kl divergence between two gaussians (Q || P)
        Parameters:
             * mu_q: torch.Tensor -> mu parameter of distribution Q
             * sigma_q: torch.Tensor -> sigma parameter of distribution Q
             * mu_p: float -> mu parameter of distribution P
             * sigma_p: float -> sigma parameter of distribution P
        returns torch.Tensor of shape 0
        """
        kl = (
            torch.log(sigma_p)
            - torch.log(sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * (sigma_p ** 2))
            - 0.5
        )

        if reduction == "sum":
            return kl.sum()
        elif reduction == "mean":
            return kl.mean()
        elif reduction == "none" or not reduction:
            return kl


class LinearFlipout(BaseVariationalLayer_):
    def __init__(
        self,
        in_features,
        out_features,
        prior_mean=0,
        prior_variance=1,
        prior_variance_scaling=None,
        posterior_mu_init=0,
        posterior_rho_init=-3.0,
        bias=True,
        default_kl_loss_reduction="sum",
    ):
        """
        Implements Linear layer with Flipout reparameterization trick.
        Ref: https://arxiv.org/abs/1803.04386
        Inherits from bayesian_torch.layers.BaseVariationalLayer_
        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.prior_mean = prior_mean

        if prior_variance_scaling is None or prior_variance_scaling == "none":
            self.prior_variance = prior_variance
        elif prior_variance_scaling == "infinite_width_limit":
            self.prior_variance = (
                prior_variance / self.in_features
            )  # as the number of input features grows to infinity, the variance of the output is now bounded under this scaling, ensuring that the GP limit exists

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer(
            "eps_weight", torch.Tensor(out_features, in_features), persistent=False
        )
        self.register_buffer(
            "prior_weight_mu", torch.Tensor(out_features, in_features), persistent=False
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(out_features, in_features),
            persistent=False,
        )

        if bias:
            self.mu_bias = nn.Parameter(torch.Tensor(out_features))
            self.rho_bias = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(out_features), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(out_features), persistent=False
            )
            self.register_buffer(
                "eps_bias", torch.Tensor(out_features), persistent=False
            )

        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None, persistent=False)

        self.init_parameters()
        self.default_kl_loss_reduction = default_kl_loss_reduction

    def init_parameters(self):
        # init prior mu
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        # init weight and base perturbation weights
        self.mu_weight.data.normal_(mean=self.posterior_mu_init, std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init, std=0.1)

        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init, std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init, std=0.1)

    def kl_loss(self, reduction=None):
        if not reduction:
            reduction = self.default_kl_loss_reduction

        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
            reduction=reduction,
        )
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
                reduction=reduction,
            )
        return kl

    def compute_weight_bias_kls(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl_w = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma,
            reduction="none",
        ).flatten()
        kl_b = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl_b = self.kl_div(
                self.mu_bias,
                sigma_bias,
                self.prior_bias_mu,
                self.prior_bias_sigma,
                reduction="none",
            ).flatten()

        return kl_w, kl_b

    def forward(self, x, mode="sample", return_kl=True, kl_reduction=None):
        if kl_reduction is None:
            kl_reduction = self.default_kl_loss_reduction

        if self.dnn_to_bnn_flag:
            return_kl = False
        # sampling delta_W
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        self.eps_weight.data.normal_()
        delta_weight = sigma_weight * self.eps_weight

        # get kl divergence
        if return_kl:
            kl = self.kl_div(
                self.mu_weight,
                sigma_weight,
                self.prior_weight_mu,
                self.prior_weight_sigma,
                reduction=kl_reduction,
            )

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = sigma_bias * self.eps_bias.data.normal_()
            if return_kl:
                kl = kl + self.kl_div(
                    self.mu_bias,
                    sigma_bias,
                    self.prior_bias_mu,
                    self.prior_bias_sigma,
                    reduction=kl_reduction,
                )

        # linear outputs
        outputs = F.linear(x, self.mu_weight, self.mu_bias)

        sign_input = x.clone().uniform_(-1, 1).sign()
        sign_output = outputs.clone().uniform_(-1, 1).sign()

        perturbed_outputs = F.linear(x * sign_input, delta_weight, bias) * sign_output

        # returning outputs + perturbations
        if return_kl:
            return outputs + perturbed_outputs, kl
        return outputs + perturbed_outputs
