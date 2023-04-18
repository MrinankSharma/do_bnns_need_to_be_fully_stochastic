# AISTATS Instructions

## 1D Regression with VI
To install dependencies, you can use poetry with the include `pyproject.toml` file, or use singularity with the definition file in the `singularity_folder`. The experiments were primarily run using the singularity definition file in the `singularity` folde.

For plotting, see `notebooks/vi_pathology_plots.ipynb`. Since the trained checkpoints are included, the notebook will run.

To produce the checkpoints, you can run the following command. This will launch multiple runs, using Hydra.
```bash
python run.py -m callbacks=toy_regression model=mixed_stochastic_mlp logger.wandb.project=1d_vi trainer.max_epochs=12000 trainer.min_epochs=12000 datamodule=toy_regression_a +model.n_mc_train=1 base_outdir=$OUTPUT_DIR model.stochastic_layer_code=8,15 seed=1,2,3,4
```

Thanks to https://github.com/ashleve/lightning-hydra-template, taken hydra with pytorch ligthning components from here.

## 1D Regression with HMC
Please see `notebooks/1d_hmc.ipynb`