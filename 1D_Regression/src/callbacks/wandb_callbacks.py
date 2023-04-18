import subprocess
from pathlib import Path

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.plotting.one_d_regression import plot_mean
from src.plotting.models import plot_kl_dict
from src.models.bayesian_mlps import compute_kl_distributions

from src import utils

log = utils.get_logger(__name__)

import multiprocessing
import time


def _start_process_w_timeout(process, timeout):
    process.start()
    process.join(30)  # 30 second timeout
    if process.is_alive():
        log.warning("Could not upload image within 30 seconds; killing process")
        process.kill()


def _create_image_upload_process(logger, key, image):
    log.debug(f"Create image upload process")

    def upload_image():
        logger.log_image(key, [image])

    return upload_image()


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class AddModelMetadataToConfig(Callback):
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        experiment.config["model_name"] = getattr(
            pl_module, "model_name", "unknown_model"
        )
        experiment.config["n_layers_stochastic"] = getattr(
            pl_module, "n_layers_stochastic", "unknown"
        )


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        try:
            if self.use_git:
                # get .git folder path
                git_dir_path = Path(
                    subprocess.check_output(["git", "rev-parse", "--git-dir"])
                    .strip()
                    .decode("utf8")
                ).resolve()

                for path in Path(self.code_dir).resolve().rglob("*"):

                    # don't upload files ignored by git
                    # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                    command = ["git", "check-ignore", "-q", str(path)]
                    not_ignored = subprocess.run(command).returncode == 1

                    # don't upload files from .git folder
                    not_git = not str(path).startswith(str(git_dir_path))

                    if path.is_file() and not_git and not_ignored:
                        code.add_file(
                            str(path), name=str(path.relative_to(self.code_dir))
                        )

            else:
                for path in Path(self.code_dir).resolve().rglob("*.py"):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

            experiment.log_artifact(code)
        except:
            log.warning("Failed to upload code to weights and biases")


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogToyRegressionPredictions(Callback):
    def __init__(self, n_mc=100, n_grid=200):
        self.n_mc = n_mc
        self.n_grid = n_grid
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            plt_small = plot_mean(
                pl_module,
                trainer.datamodule,
                n_mc=self.n_mc,
                n_grid=self.n_grid,
                title=f"Epoch: {trainer.current_epoch}\n{pl_module.model_name}",
                xlim=trainer.datamodule.lim_small["x"],
                ylim=trainer.datamodule.lim_small["y"],
            )

            plt_large = plot_mean(
                pl_module,
                trainer.datamodule,
                n_mc=self.n_mc,
                n_grid=self.n_grid,
                title=f"Epoch: {trainer.current_epoch}\n{pl_module.model_name}",
                xlim=trainer.datamodule.lim_large["x"],
                ylim=trainer.datamodule.lim_large["y"],
            )

            p1 = multiprocessing.Process(
                target=_create_image_upload_process(
                    logger, "Images/mean_predictions_small", plt_small
                )
            )
            p2 = multiprocessing.Process(
                target=_create_image_upload_process(
                    logger, "Images/mean_predictions_large", plt_large
                )
            )

            _start_process_w_timeout(p1, 30)
            _start_process_w_timeout(p2, 30)


class LogLayerwiseKLDistributions(Callback):
    def __init__(self):
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment  # actual wandb object
            kl_dict = compute_kl_distributions(pl_module)

            if len(kl_dict) > 0:
                plt = plot_kl_dict(
                    kl_dict,
                    title=f"Epoch: {trainer.current_epoch}\n{pl_module.model_name}",
                )

                p = multiprocessing.Process(
                    target=_create_image_upload_process(
                        logger, "Images/layerwise_kls", plt
                    )
                )

                _start_process_w_timeout(p, 30)
