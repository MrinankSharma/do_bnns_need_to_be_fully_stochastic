import numpy as np
from pl_bolts.datamodules import SklearnDataModule


class ToyRegressionADataModule(SklearnDataModule):
    lim_small = {"x": (-3.5, 3.5), "y": (-1.25, 1.25)}
    lim_large = {"x": (-5, 5), "y": (-3, 3)}

    def __init__(
        self,
        n_points_bin=1000,
        output_noise_scale=0.05,
        *args,
        **kwargs,
    ) -> None:
        """
        Toy regression dataset taken from https://arxiv.org/pdf/2002.03704.pdf

        :param n_points_bin: points in each bin to sample
        :param output_noise_scale: Gaussian output noise scale
        :param args: Arguments passed to SklearnDataModule
        :param kwargs: Arguments passed to SklearnDataModule
        """
        random_state = np.random.get_state()  # store current state
        np.random.seed(1)  # manually fix seed when generating random dataset
        X = np.hstack(
            [
                np.random.uniform(-2, -1.4, n_points_bin),
                np.random.uniform(2.0, 2.8, n_points_bin),
            ]
        )
        y = np.sin(4 * (X - 4.3))
        y = y + (output_noise_scale * np.random.normal(size=y.shape))
        np.random.set_state(random_state)

        nD = X.size
        X = X.reshape((nD, 1))
        y = y.reshape((nD, 1))

        super().__init__(X, y, *args, **kwargs)

class ToyRegressionDDataModule(SklearnDataModule):
    lim_small = {"x": (-3.5, 3.5), "y": (-1.25, 1.25)}
    lim_large = {"x": (-5, 5), "y": (-3, 3)}

    def __init__(
        self,
        n_points_bin=25,
        output_noise_scale=0.05,
        *args,
        **kwargs,
    ) -> None:
        """
        Toy regression dataset taken from https://arxiv.org/pdf/2002.03704.pdf

        :param n_points_bin: points in each bin to sample
        :param output_noise_scale: Gaussian output noise scale
        :param args: Arguments passed to SklearnDataModule
        :param kwargs: Arguments passed to SklearnDataModule
        """
        random_state = np.random.get_state()  # store current state
        np.random.seed(1)  # manually fix seed when generating random dataset
        X = np.hstack(
            [
                np.linspace(-3, -1.7, n_points_bin),
                np.linspace(2.2, 4, n_points_bin),
            ]
        )
        y = np.sin(4 * (X - 4.3))
        y = y + (output_noise_scale * np.random.normal(size=y.shape))
        np.random.set_state(random_state)

        nD = X.size
        X = X.reshape((nD, 1))
        y = y.reshape((nD, 1))

        super().__init__(X, y, *args, **kwargs)