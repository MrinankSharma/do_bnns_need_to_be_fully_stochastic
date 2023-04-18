# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide ResNet with variational Bayesian layers."""
import functools
from typing import Dict, Iterable, Optional

from absl import logging
import numpy as np
import tensorflow as tf

_HP_KEYS = (
    "bn_l2",
    "input_conv_l2",
    "group_1_conv_l2",
    "group_2_conv_l2",
    "group_3_conv_l2",
    "dense_kernel_l2",
    "dense_bias_l2",
)

try:
    import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError:
    logging.warning("Skipped edward2 import due to ImportError.", exc_info=True)

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9,
)
Conv2DFlipout = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DFlipout, kernel_size=3, padding="same", use_bias=False
)


def Conv2D(filters, seed=None, **kwargs):  # pylint: disable=invalid-name
    """Conv2D layer that is deterministically initialized."""
    default_kwargs = {
        "kernel_size": 3,
        "padding": "same",
        "use_bias": False,
        # Note that we need to use the class constructor for the initializer to
        # get deterministic initialization.
        "kernel_initializer": tf.keras.initializers.HeNormal(seed=seed),
    }
    # Override defaults with the passed kwargs.
    default_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(filters, **default_kwargs)


def deterministic_basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int,
    version: int = 2,
) -> tf.Tensor:
    """Basic residual block of two 3x3 convs.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      strides: Stride dimensions for Conv2D.
      conv_l2: L2 regularization coefficient for the conv kernels.
      bn_l2: L2 regularization coefficient for the batch norm layers.
      seed: random seed used for initialization.
      version: 1, indicating the original ordering from He et al. (2015); or 2,
        indicating the preactivation ordering from He et al. (2016).

    Returns:
      tf.Tensor.
    """
    x = inputs
    y = inputs
    if version == 2:
        y = BatchNormalization(
            beta_regularizer=tf.keras.regularizers.l2(bn_l2),
            gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
        )(y)
        y = tf.keras.layers.Activation("relu")(y)
    seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]
    y = Conv2D(
        filters,
        strides=strides,
        seed=seeds[0],
        kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
    )(y)
    y = BatchNormalization(
        beta_regularizer=tf.keras.regularizers.l2(bn_l2),
        gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
    )(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = Conv2D(
        filters,
        strides=1,
        seed=seeds[1],
        kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
    )(y)
    if version == 1:
        y = BatchNormalization(
            beta_regularizer=tf.keras.regularizers.l2(bn_l2),
            gamma_regularizer=tf.keras.regularizers.l2(bn_l2),
        )(y)
    if not x.shape.is_compatible_with(y.shape):
        x = Conv2D(
            filters,
            kernel_size=1,
            strides=strides,
            seed=seeds[2],
            kernel_regularizer=tf.keras.regularizers.l2(conv_l2),
        )(x)
    x = tf.keras.layers.add([x, y])
    if version == 1:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def stochastic_basic_block(
    inputs, filters, strides, prior_stddev, dataset_size, stddev_init
):
    """Basic residual block of two 3x3 convs.

    Args:
      inputs: tf.Tensor.
      filters: Number of filters for Conv2D.
      strides: Stride dimensions for Conv2D.
      prior_stddev: Fixed standard deviation for weight prior.
      dataset_size: Dataset size to properly scale the KL.
      stddev_init: float to initialize variational posterior stddev parameters.

    Returns:
      tf.Tensor.
    """
    x = inputs
    y = inputs
    y = BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = Conv2DFlipout(
        filters,
        strides=strides,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_init)), stddev=0.1
            )
        ),
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1.0 / dataset_size
        ),
    )(y)
    y = BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = Conv2DFlipout(
        filters,
        strides=1,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_init)), stddev=0.1
            )
        ),
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1.0 / dataset_size
        ),
    )(y)
    if not x.shape.is_compatible_with(y.shape):
        x = Conv2DFlipout(
            filters,
            kernel_size=1,
            strides=strides,
            kernel_initializer=ed.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(stddev_init)), stddev=0.1
                )
            ),
            kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
                stddev=prior_stddev, scale_factor=1.0 / dataset_size
            ),
        )(x)
    x = tf.keras.layers.add([x, y])
    return x


def group(
    inputs,
    filters,
    strides,
    num_blocks,
    seed,
    conv_l2,
    bn_l2,
    version,
    stochastic_group_flags,
    **kwargs,
):
    """Group of residual blocks."""
    seeds = tf.random.experimental.stateless_split([seed, seed + 1], num_blocks)[:, 0]
    if stochastic_group_flags[0]:
        x = stochastic_basic_block(inputs, filters=filters, strides=strides, **kwargs)
    else:
        x = deterministic_basic_block(
            inputs,
            filters=filters,
            strides=strides,
            conv_l2=conv_l2,
            bn_l2=bn_l2,
            version=version,
            seed=seeds[0],
        )

    for i in range(num_blocks - 1):
        if stochastic_group_flags[i + 1]:
            x = stochastic_basic_block(x, filters=filters, strides=1, **kwargs)
        else:
            x = deterministic_basic_block(
                x,
                filters=filters,
                strides=1,
                conv_l2=conv_l2,
                bn_l2=bn_l2,
                version=version,
                seed=seeds[i + 1],
            )
    return x


def _parse_hyperparameters(l2: float, hps: Dict[str, float]):
    """Extract the L2 parameters for the dense, conv and batch-norm layers."""

    assert_msg = (
        "Ambiguous hyperparameter specifications: either l2 or hps "
        "must be provided (received {} and {}).".format(l2, hps)
    )
    is_specified = lambda h: bool(h) and all(v is not None for v in h.values())
    only_l2_is_specified = l2 is not None and not is_specified(hps)
    only_hps_is_specified = l2 is None and is_specified(hps)
    assert only_l2_is_specified or only_hps_is_specified, assert_msg
    if only_hps_is_specified:
        assert_msg = "hps must contain the keys {}!={}.".format(_HP_KEYS, hps.keys())
        assert set(hps.keys()).issuperset(_HP_KEYS), assert_msg
        return hps
    else:
        return {k: l2 for k in _HP_KEYS}


def partially_stochastic_variational_wide_resnet(
    input_shape,
    depth,
    width_multiplier,
    num_classes,
    prior_stddev,
    dataset_size,
    stddev_init,
    l2: float,
    stochastic_layer_code,
    seed: int = 42,
    hps: Optional[Dict[str, float]] = None,
):
    """Builds Wide ResNet.

    Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
    number of filters. Using three groups of residual blocks, the network maps
    spatial features of size 32x32 -> 16x16 -> 8x8.

    Args:
      input_shape: tf.Tensor.
      depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
        He et al. (2015)'s notation which uses the maximum depth of the network
        counting non-conv layers like dense.
      width_multiplier: Integer to multiply the number of typical filters by. "k"
        in WRN-n-k.
      num_classes: Number of output classes.
      prior_stddev: Fixed standard deviation for weight prior.
      dataset_size: Dataset size to properly scale the KL.
      stddev_init: float to initialize variational posterior stddev parameters.

    Returns:
      tf.keras.Model.
    """
    if (depth - 4) % 6 != 0:
        raise ValueError("depth should be 6n+4 (e.g., 16, 22, 28, 40).")
    num_blocks = (depth - 4) // 6
    inputs = tf.keras.layers.Input(shape=input_shape)

    seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]

    assert stochastic_layer_code > -1
    assert stochastic_layer_code < 2 ** 14

    stochastic_layer_flags = [stochastic_layer_code & 2 ** i > 0 for i in range(0, 14)]

    name_str = ""
    for l_i, f in enumerate(stochastic_layer_flags):
        if l_i == 0:
            name_str = f"Input: {'Stoc' if f else 'Det'}"
        elif l_i < 13:
            name_str = f"{name_str}\nResnetBlock-{l_i}: {'Stoc' if f else 'Det'}"
        elif l_i == 13:
            name_str = f"{name_str}\nOutput-{l_i}: {'Stoc' if f else 'Det'}"

    print(f"Stochastic layer code {stochastic_layer_code} converted into MixedStochasticWideResnet::\n{name_str}")

    l2_reg = tf.keras.regularizers.l2
    hps = _parse_hyperparameters(l2, hps)

    if stochastic_layer_flags[0]:
        x = Conv2DFlipout(
            16,
            strides=1,
            kernel_initializer=ed.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(stddev_init)), stddev=0.1
                )
            ),
            kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
                stddev=prior_stddev, scale_factor=1.0 / dataset_size
            ),
        )(inputs)
    else:
        x = Conv2D(
            16,
            strides=1,
            seed=seeds[0],
            kernel_regularizer=l2_reg(hps["input_conv_l2"]),
        )(inputs)

    x = group(
        x,
        filters=16 * width_multiplier,
        strides=1,
        num_blocks=num_blocks,
        conv_l2=hps["group_1_conv_l2"],
        bn_l2=hps["bn_l2"],
        version=2,
        stochastic_group_flags=stochastic_layer_flags[1 : 1 + num_blocks],
        seed=seeds[1],
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_init=stddev_init
    )
    x = group(
        x,
        filters=32 * width_multiplier,
        strides=2,
        num_blocks=num_blocks,
        conv_l2=hps["group_2_conv_l2"],
        bn_l2=hps["bn_l2"],
        version=2,
        stochastic_group_flags=stochastic_layer_flags[
            1 + num_blocks : 1 + 2 * num_blocks
        ],
        seed=seeds[2],
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_init=stddev_init
    )
    x = group(
        x,
        filters=64 * width_multiplier,
        strides=2,
        num_blocks=num_blocks,
        conv_l2=hps["group_3_conv_l2"],
        bn_l2=hps["bn_l2"],
        version=2,
        stochastic_group_flags=stochastic_layer_flags[
            1 + 2 * num_blocks : 1 + 3 * num_blocks
        ],
        seed=seeds[3],
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_init=stddev_init
    )

    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)

    if stochastic_layer_flags[-1]:
        x = ed.layers.DenseFlipout(
            num_classes,
            kernel_initializer=ed.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(stddev_init)), stddev=0.1
                )
            ),
            kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
                stddev=prior_stddev, scale_factor=1.0 / dataset_size
            ),
        )(x)
    else:
        x = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
            kernel_regularizer=l2_reg(hps["dense_kernel_l2"]),
            bias_regularizer=l2_reg(hps["dense_bias_l2"]),
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
