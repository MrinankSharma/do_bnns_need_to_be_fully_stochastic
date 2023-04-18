import sys, os, time, requests

sys.path.append(os.getcwd())

from functools import partial

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, Trace_ELBO, autoguide, SVI

from numpyro.infer.initialization import init_to_uniform

import jax
import jax.numpy as jnp
import jax.nn
from jax import random

numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--seed", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--gap", action="store_true")
parser.add_argument("--prior_variance", type=float, default=0.1) #0.1 is good for yacht, but not for other datasets
parser.add_argument("--likelihood_scale", type=float, default=6.0) #6.0 is good for yacht, but not for other datasets


def _gap_train_test_split(X, y, gap_column, test_size):
    n_data = X.shape[0]
    sorted_idxs = np.argsort(X[:, gap_column])
    train_idxs = np.concatenate(
        (
            sorted_idxs[: int(n_data * 0.5 * (1 - test_size))],
            sorted_idxs[-int(n_data * 0.5 * (1 - test_size)) :],
        )
    )
    test_idxs = np.array(list(set(sorted_idxs.tolist()) - set(train_idxs.tolist())))
    X_train = X[train_idxs, :]
    X_test = X[test_idxs, :]
    y_train = y[train_idxs, :]
    y_test = y[test_idxs, :]
    return X_train, X_test, y_train, y_test


### dataset stuff


class UCIDataset:
    def __init__(
        self,
        data_dir,
        test_split_type="random",
        gap_column=0,
        test_size=0.2,
        val_fraction_of_train=0.1,
        seed=42,
        *args,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.download()

        random_state = np.random.get_state()  # store current state
        X, y = self.load_from_filepath()

        np.random.seed(seed)
        if gap_column == "random":
            gap_column = np.random.randint(0, X.shape[1])

        if test_split_type == "random":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )
        elif test_split_type == "gap":
            assert gap_column in list(range(X.shape[1]))
            X_train, X_test, y_train, y_test = _gap_train_test_split(
                X, y, gap_column, test_size
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_fraction_of_train
        )

        self.scl_X = StandardScaler()
        self.scl_X.fit(X_train)
        X_train, X_val, X_test = (
            self.scl_X.transform(X_train),
            self.scl_X.transform(X_val),
            self.scl_X.transform(X_test),
        )

        self.scl_Y = StandardScaler()
        self.scl_Y.fit(y_train)
        y_train, y_val, y_test = (
            self.scl_Y.transform(y_train),
            self.scl_Y.transform(y_val),
            self.scl_Y.transform(y_test),
        )

        np.random.set_state(random_state)

        n_train, n_val, n_test = y_train.shape[0], y_val.shape[0], y_test.shape[0]
        n_total = n_train + n_val + n_test

        print(
            f"Train set: {n_train} examples, {100*(n_train/n_total):.2f}% of all examples."
        )
        print(
            f"Val set: {n_val} examples, {100 * (n_val / n_total):.2f}% of all examples."
        )
        print(
            f"Test set: {n_test} examples, {100 * (n_test / n_total):.2f}% of all examples."
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    @property
    def file_path(self):
        return os.path.join(
            self.data_dir, "uci_datasets", self.dataset_name, self.filename
        )

    def download(self):
        if not os.path.isfile(self.file_path):
            print(f"Downloading {self.dataset_name} Dataset")
            downloaded_file = requests.get(self.url)
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "wb") as f:
                print(f"Writing {self.dataset_name} Dataset")
                f.write(downloaded_file.content)
        else:
            print(f"{self.dataset_name} Dataset already downloaded; skipping download.")


class UCIYachtDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    filename = "yacht.data"
    dataset_name = "yacht"

    def load_from_filepath(self):
        df = pd.read_fwf(self.file_path, header=None)

        X = df.values[:-1, :-1]
        nD = X.shape[0]
        y = df.values[:-1, -1].reshape((nD, 1))

        return X, y


class UCIEnergyDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    filename = "energy.xlsx"
    dataset_name = "energy"

    def load_from_filepath(self):
        df = pd.read_excel(self.file_path, engine="openpyxl")

        X = df[["X1", "X2", "X3", "X4", "X5", "X7", "X6", "X8"]].values
        nD = X.shape[0]
        y = df[["Y1"]].values.reshape((nD, 1))

        return X, y

class UCIBostonDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = "housing.data"
    dataset_name = "boston"

    def load_from_filepath(self):
        df = pd.read_fwf(self.file_path, header=None)

        X = df.values[:-1, :-1]
        nD = X.shape[0]
        y = df.values[:-1, -1].reshape((nD, 1))

        return X, y

class UCIConcreteDataset(UCIDataset):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    filename = "concrete.xls"
    dataset_name = "concrete"

    def load_from_filepath(self):
        df = pd.read_excel(self.file_path, engine="xlrd")

        X = df.values[:, :-1]
        nD = X.shape[0]
        y = df.values[:, -1].reshape((nD, 1))

        return X, y


def one_d_bnn(X, y=None, prior_variance=0.1, width=50, scale=1.0):
    nB, n_features = X.shape

    W_1 = numpyro.sample(
        "W1", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((n_features, width)))
    )
    b_1 = numpyro.sample(
        "b1", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
    )

    W_2 = numpyro.sample(
        "W2", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, width)))
    )
    b_2 = numpyro.sample(
        "b2", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
    )

    W_output = numpyro.sample(
        "W_output", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1)))
    )
    b_output = numpyro.sample(
        "b_output", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((1, 1)))
    )

    z1 = X @ W_1 + b_1.reshape((1, width)).repeat(nB, axis=0)
    h1 = jax.nn.leaky_relu(z1)

    z2 = h1 @ W_2 + b_2.reshape((1, width)).repeat(nB, axis=0)
    h2 = jax.nn.leaky_relu(z2)

    output = h2 @ W_output + b_output.repeat(nB, axis=0)
    mean = numpyro.deterministic("mean", output)

    # output precision
    prec_obs = numpyro.sample(
        "prec_obs", dist.Gamma(3.0, 1.0)
    )
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    with numpyro.handlers.scale(scale=scale):
        y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)


def evaluate_samples(model, rng_key, X, y, samples, y_scale=1.0, y_loc=0.0):
    sigma_obs = (1.0 / jnp.sqrt(samples["prec_obs"])).mean()

    predictive = Predictive(model, samples)(rng_key, X=X)

    predictive_mean = (y_scale * predictive["mean"].mean(axis=0)) + y_loc
    log_likelihood = (
        dist.Normal(predictive_mean, y_scale * sigma_obs)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )
    rmse = (
        (y_scale * (predictive["mean"].mean(axis=0).flatten() - y.flatten())) ** 2
    ).mean() ** 0.5

    return float(log_likelihood), float(rmse)


def evaluate_MAP(model, svi_results, X, y, rng_key, y_scale=1.0, y_loc=0.0):
    predictive = Predictive(
        model=model,
        guide=autoguide.AutoDelta(model),
        params=svi_results.params,
        num_samples=1,
    )(rng_key, X=X)

    sigma_obs = 1.0 / jnp.sqrt(svi_results.params["prec_obs_auto_loc"])
    rmse = (((predictive["mean"][0, :] - y) ** 2).mean() ** 0.5) * y_scale
    log_likelihood = (
        dist.Normal((y_scale * predictive["mean"][0, :]) + y_loc, sigma_obs * y_scale)
        .log_prob(y_loc + (y * y_scale))
        .mean()
    )

    return float(log_likelihood), float(rmse)


def generate_mixed_bnn_by_param(
    MAP_params, sample_mask_tuple, prior_variance, scale=1.0
):
    (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask,
    ) = sample_mask_tuple

    def mixed_bnn(X, y=None, prior_variance=prior_variance, width=50, scale=scale):
        nB, n_features = X.shape

        W_1_noise = numpyro.sample(
            "W1_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((n_features, width))),
        )
        b_1_noise = numpyro.sample(
            "b1_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
        )
        W_2_noise = numpyro.sample(
            "W2_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, width))),
        )
        b_2_noise = numpyro.sample(
            "b2_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width)))
        )
        W_output_noise = numpyro.sample(
            "W_output_noise",
            dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((width, 1))),
        )
        b_output_noise = numpyro.sample(
            "b_output_noise", dist.Normal(0, (prior_variance ** 0.5) * jnp.ones((1, 1)))
        )

        W_1_map = MAP_params["W1_auto_loc"]
        b_1_map = MAP_params["b1_auto_loc"]
        W_2_map = MAP_params["W2_auto_loc"]
        b_2_map = MAP_params["b2_auto_loc"]
        W_output_map = MAP_params["W_output_auto_loc"]
        b_output_map = MAP_params["b_output_auto_loc"]

        W_1 = numpyro.deterministic(
            "W1", (W_1_map * (1 - W1_sample_mask)) + (W_1_noise * W1_sample_mask)
        )
        W_2 = numpyro.deterministic(
            "W2", (W_2_map * (1 - W2_sample_mask)) + (W_2_noise * W2_sample_mask)
        )
        W_output = numpyro.deterministic(
            "W_output",
            (W_output_map * (1 - W_output_sample_mask))
            + (W_output_noise * W_output_sample_mask),
        )

        b_1 = numpyro.deterministic(
            "b1", (b_1_map * (1 - b1_sample_mask)) + (b_1_noise * b1_sample_mask)
        )
        b_2 = numpyro.deterministic(
            "b2", (b_2_map * (1 - b2_sample_mask)) + (b_2_noise * b2_sample_mask)
        )
        b_output = numpyro.deterministic(
            "b_output",
            (b_output_map * (1 - b_output_sample_mask))
            + (b_output_noise * b_output_sample_mask),
        )

        z1 = X @ W_1 + b_1.reshape((1, width)).repeat(nB, axis=0)
        h1 = jax.nn.leaky_relu(z1)

        z2 = h1 @ W_2 + b_2.reshape((1, width)).repeat(nB, axis=0)
        h2 = jax.nn.leaky_relu(z2)

        output = h2 @ W_output + b_output.repeat(nB, axis=0)
        mean = numpyro.deterministic("mean", output)

        # output precision
        prec_obs = numpyro.sample(
            "prec_obs", dist.Gamma(3.0, 1.0)
        )  # MAP outperforms full BNN, even if we freeze the prior precision. That's interesting here, I think.
        sigma_obs = 1.0 / jnp.sqrt(prec_obs)

        with numpyro.handlers.scale(scale=scale):
            y_obs = numpyro.sample("y_obs", dist.Normal(mean, sigma_obs), obs=y)

    return mixed_bnn


def create_sample_mask_largest_abs_values(percentile, MAP_params):
    keys = [
        "W1_auto_loc",
        "W2_auto_loc",
        "W_output_auto_loc",
        "b1_auto_loc",
        "b2_auto_loc",
        "b_output_auto_loc",
    ]

    all_values = np.concatenate([MAP_params[key].ravel() for key in keys])
    param_abs_values = np.abs(all_values)
    val = np.percentile(param_abs_values, 100 - percentile)

    W1_sample_mask = np.abs(MAP_params["W1_auto_loc"]) >= val
    W2_sample_mask = np.abs(MAP_params["W2_auto_loc"]) >= val
    W_output_sample_mask = np.abs(MAP_params["W_output_auto_loc"]) >= val
    b1_sample_mask = np.abs(MAP_params["b1_auto_loc"]) >= val
    b2_sample_mask = np.abs(MAP_params["b2_auto_loc"]) >= val
    b_output_sample_mask = np.abs(MAP_params["b_output_auto_loc"]) >= val

    sample_mask_tuple = (
        W1_sample_mask,
        W2_sample_mask,
        W_output_sample_mask,
        b1_sample_mask,
        b2_sample_mask,
        b_output_sample_mask,
    )

    return sample_mask_tuple


def run_for_percentile(
    dataset,
    percentile,
    MAP_params,
    prior_variance=0.8,
    prior_variance_scaled=True,
    scale=1.0,
):
    sample_mask_tuple = create_sample_mask_largest_abs_values(percentile, MAP_params)
    prior_variance_used = (
        prior_variance
        if not prior_variance_scaled
        else prior_variance * (100 / percentile)
    )

    mixed_bnn = generate_mixed_bnn_by_param(
        MAP_params,
        create_sample_mask_largest_abs_values(percentile, MAP_params),
        prior_variance_used,
        scale=scale,
    )

    nuts_kernel = NUTS(mixed_bnn, max_tree_depth=15)
    mcmc = MCMC(nuts_kernel, num_warmup=325, num_samples=75, num_chains=8)
    rng_key = random.PRNGKey(0)

    start_time = time.time()
    mcmc.run(rng_key, dataset.X_train, dataset.y_train)
    end_time = time.time()

    train_ll, train_rmse = evaluate_samples(
        mixed_bnn,
        rng_key,
        dataset.X_train,
        dataset.y_train,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    val_ll, val_rmse = evaluate_samples(
        mixed_bnn,
        rng_key,
        dataset.X_val,
        dataset.y_val,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    test_ll, test_rmse = evaluate_samples(
        mixed_bnn,
        rng_key,
        dataset.X_test,
        dataset.y_test,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )

    results = {
        "prior_variance": prior_variance_used,
        "test_rmse": test_rmse,
        "test_ll": test_ll,
        "val_rmse": val_rmse,
        "val_ll": val_ll,
        "train_rmse": train_rmse,
        "train_ll": train_ll,
        "runtime": end_time - start_time,
        "num_params_sampled": np.array([t.sum() for t in sample_mask_tuple]).sum(),
        "dataset": args.dataset,
        "seed": args.seed,
        "gap_split?": args.gap,
        "prior_variance_scaled": True,
        "name": f"percent_sampled_{percentile:.2f}",
        "scale": scale,
    }

    return results


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset == "yacht":
        dataset_class = UCIYachtDataset
    elif args.dataset == "energy":
        dataset_class = UCIEnergyDataset
    elif args.dataset == "concrete":
        dataset_class = UCIConcreteDataset
    elif args.dataset == "boston":
        dataset_class = UCIBostonDataset

    dataset = dataset_class(
        "/data/uci_datasets",
        gap_column="random",
        seed=args.seed,
        test_split_type="gap" if args.gap else "random",
        test_size=0.1,
        val_fraction_of_train=0.1,
    )

    ### Train MAP Solution
    optimizer = numpyro.optim.Adam(0.01)
    rng_key = random.PRNGKey(1)

    model = lambda X, y=None: one_d_bnn(X, y, prior_variance=args.prior_variance)

    svi = SVI(model, autoguide.AutoDelta(one_d_bnn), optimizer, Trace_ELBO())
    start_time = time.time()
    svi_results = svi.run(rng_key, 20000, X=dataset.X_train, y=dataset.y_train)
    end_time = time.time()

    train_ll, train_rmse = evaluate_MAP(
        model,
        svi_results,
        dataset.X_train,
        dataset.y_train,
        rng_key,
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    val_ll, val_rmse = evaluate_MAP(
        model,
        svi_results,
        dataset.X_val,
        dataset.y_val,
        rng_key,
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    test_ll, test_rmse = evaluate_MAP(
        model,
        svi_results,
        dataset.X_test,
        dataset.y_test,
        rng_key,
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )

    map_results = {
        "prior_variance": args.prior_variance,
        "test_rmse": test_rmse,
        "test_ll": test_ll,
        "val_rmse": val_rmse,
        "val_ll": val_ll,
        "train_rmse": train_rmse,
        "train_ll": train_ll,
        "runtime": end_time - start_time,
        "num_params_sampled": 0,
        "dataset": args.dataset,
        "seed": args.seed,
        "gap_split?": args.gap,
        "name": "MAP",
    }

    print(map_results)

    ### Train Full HMC results
    model = lambda X, y=None: one_d_bnn(
        X, y, prior_variance=args.prior_variance, scale=args.likelihood_scale
    )
    nuts_kernel = NUTS(model, max_tree_depth=15)
    mcmc = MCMC(nuts_kernel, num_warmup=325, num_samples=75, num_chains=8)
    rng_key = random.PRNGKey(0)

    start_time = time.time()
    mcmc.run(rng_key, dataset.X_train, dataset.y_train)
    end_time = time.time()

    train_ll, train_rmse = evaluate_samples(
        one_d_bnn,
        rng_key,
        dataset.X_train,
        dataset.y_train,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    val_ll, val_rmse = evaluate_samples(
        one_d_bnn,
        rng_key,
        dataset.X_val,
        dataset.y_val,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )
    test_ll, test_rmse = evaluate_samples(
        one_d_bnn,
        rng_key,
        dataset.X_test,
        dataset.y_test,
        mcmc.get_samples(),
        y_scale=dataset.scl_Y.scale_,
        y_loc=dataset.scl_Y.mean_,
    )

    full_network_results = {
        "prior_variance": args.prior_variance,
        "test_rmse": test_rmse,
        "test_ll": test_ll,
        "val_rmse": val_rmse,
        "val_ll": val_ll,
        "train_rmse": train_rmse,
        "train_ll": train_ll,
        "runtime": end_time - start_time,
        "num_params_sampled": 2951,
        "dataset": args.dataset,
        "seed": args.seed,
        "gap_split?": args.gap,
        "name": "full_network",
        "scale": args.likelihood_scale,
    }

    print(full_network_results)

    percentiles = list(np.logspace(-1, 1.996, 15))
    MAP_params = svi_results.params

    all_results = {
        "all_results_not_scaled": [map_results, full_network_results],
        "all_results_scaled": [map_results, full_network_results],
    }

    fname = f"/data/uci_subset_hmc/{args.dataset}_s{args.seed}_prior_var{args.prior_variance:.2f}_scale{args.likelihood_scale}"

    if args.gap:
        fname = f"{fname}_gap"

    import pickle

    for percentile in percentiles:
        print(
            f"Running for {percentile} of weights sampled scaled, by maximum absolute value"
        )
        all_results["all_results_scaled"].append(
            run_for_percentile(
                dataset,
                percentile,
                MAP_params,
                prior_variance_scaled=True,
                scale=args.likelihood_scale,
            )
        )
        print(all_results["all_results_scaled"][-1])

        pickle.dump(all_results, open(f"{fname}.pkl", "wb"))

    for percentile in percentiles:
        print(f"Running for {percentile} of weights sampled, by maximum absolute value")
        all_results["all_results_not_scaled"].append(
            run_for_percentile(
                dataset,
                percentile,
                MAP_params,
                prior_variance_scaled=False,
                scale=args.likelihood_scale,
            )
        )
        print(all_results["all_results_not_scaled"][-1])

        pickle.dump(all_results, open(f"{fname}.pkl", "wb"))
