import gpbasics.global_parameters as global_param

global_param.ensure_init()

import os
import logging
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.optimizer.nelder_mead import NelderMeadOptimizerResults
import gpbasics.DataHandling.DataInput as di


class SequentialChangePointDetectionType(Enum):
    WhiteNoise = 1


class SequentialChangePointDetection:
    """
    This is a general class defining the structure for those implementations, which return just the next change
    point per request instead of returning all change points of a dataset at once. This notion of _Sequential
    Change Point Detection_ has a certain overlap with the concept of _Online Change Point Detection_, but we do
    not regard these two concepts as interchangeable.
    """
    def __init__(self, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]]):
        self.data_input: di.AbstractDataInput
        if isinstance(data_input, list):
            logging.warning("Divisive Gaussian process model inference algorithms currently need a representative "
                            "fold in order to adequately derive a common split of the dataset.")
            self.data_input = data_input[0]
        else:
            self.data_input = data_input

        self.absolute_minimum_segment_size = 80

    def get_next_change_point(self, start_index: int, stop_index: int) -> Tuple[bool, int]:
        """
        This method returns a change point of the dataset specified (self.data_input) within the window specified by
        the start and stop index.
        Args:
            start_index: Indicates the beginning of the considered data window
            stop_index: Indicates the end of the considered data window

        Returns: a Tuple, where the first element indicates, whether change point detection within this window was
        successfull, and the second element identifies the respective change point index.
        """
        pass


class WhiteNoiseCPD(SequentialChangePointDetection):
    def __init__(self, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]], default_window_size: int):
        super(WhiteNoiseCPD, self).__init__(data_input)
        self.default_window_size: int = default_window_size

    def get_next_change_point(self, start_index: int, stop_index: int) -> Tuple[bool, int]:
        # Initialize new change point via middle value of the considered window of the input training data
        x_train = self.data_input.data_x_train[start_index: stop_index]

        # the considered window of the input training data is normalized
        # this ensures, that optimization of the change point is not subject to a very narrow range of input training
        # data, which would impede finding an apt position for the change point.
        # This might especially benefit segmenting larger datasets.
        x_train -= tf.reduce_min(x_train)
        x_train /= tf.reduce_max(x_train)

        cp: float = x_train[(stop_index - start_index) // 2]

        # target training data normalized by subtracting mean function evaluated at training input data
        detrended_y_train = self.data_input.get_detrended_y_train()[start_index: stop_index]
        detrended_y_train -= tf.reduce_min(detrended_y_train)
        detrended_y_train /= tf.math.reduce_max(detrended_y_train)

        # Each segment is initially modelled by the same noise
        initial_noise_per_segment: float = 1 + global_param.p_cov_matrix_jitter

        # If a change point cannot be found in the given range (start_index to stop_index), 3CS / SKS restarts the
        # search for a change point with a window of an increased size. In order to prohibit indefinite search for a
        # change point, the noise of the first / left segment is slightly changed in case
        # an increased window size is used.
        noise_left = initial_noise_per_segment + \
                     (global_param.p_cov_matrix_jitter * ((stop_index - start_index) / self.default_window_size))

        param = tf.Variable(
            [noise_left, initial_noise_per_segment, tf.squeeze(cp, axis=-1)], dtype=global_param.p_dtype, shape=[3, ])

        def opt():
            return white_noise_ll([param[0], param[1]], param[2], stop_index - start_index, detrended_y_train)

        sgd_opt: tfp.optimizer.VariationalSGD = \
            tfp.optimizer.VariationalSGD(batch_size=1, total_num_examples=(stop_index - start_index))

        sgd_opt.minimize(opt, [param])

        if tf.reduce_any(tf.math.is_nan(param)):
            return False, int(stop_index)

        new_cp_index = int(param[2] * (stop_index - start_index) + start_index)

        is_same_noise = (param[0] - param[1]) <= 1e-12

        # if the found change point is too close to either one of the ends of the considered window of the
        # training data, segmentation failed (success = False) and segmentation need to be rerun with increased window.
        if is_same_noise or new_cp_index - start_index < self.absolute_minimum_segment_size \
                or stop_index - new_cp_index < 20:
            return False, int(stop_index)

        return True, int(new_cp_index)


@tf.function(experimental_relax_shapes=True)
def approx_indicator(x: tf.constant, cp: tf.Tensor) -> tf.Tensor:
    # the employed sigmoid function is very steep, thus sigmoid slope is very high
    sigmoid_slope: tf.constant = tf.constant(1000, dtype=global_param.p_dtype)
    return tf.math.divide(tf.cast(1, dtype=global_param.p_dtype),
                          tf.math.add(tf.cast(1, dtype=global_param.p_dtype),
                                      tf.math.exp(
                                          tf.cast(-1, dtype=global_param.p_dtype) * sigmoid_slope * (x - cp))))


@tf.function(experimental_relax_shapes=True)
def white_noise_ll(noises: List[tf.Tensor], cp_position: tf.Tensor, segment_length: int, detrended_y_train) \
        -> tf.Tensor:
    """
    This method calculates the likelihood of a full Gaussian process, with zero mean and a covariance functions,
    which models two separate segments of the data with two white noise kernels. The delimitation between these
    segments is controlled by a change point. Since a white noise kernel only produces a diagonal matrix, where the
    noise is on that diagonal, likelihood calcualtion is simplified substantially.
    Args:
        noises: List of two tf.Tensor, which each represent the noise of one segment
        cp_position: position of the change point, cp \in [0; 1]
        segment_length: length of the currently considered segment / window
        detrended_y_train: target training data minus the mean function evaluated at input training data of the
        considered data window

    Returns: negative log marginal likelihood

    """
    normalized_cp: tf.Tensor = cp_position

    input_x = tf.cast(tf.linspace(0, 1, segment_length), dtype=global_param.p_dtype)

    sigmoid_vector = approx_indicator(input_x, normalized_cp)

    _sigmoid_vector: tf.Tensor = tf.subtract(tf.cast(1, global_param.p_dtype), sigmoid_vector)

    diag_1: tf.Tensor = tf.multiply(tf.fill([segment_length, ], tf.divide(1, noises[0])), _sigmoid_vector)
    diag_2: tf.Tensor = tf.multiply(tf.fill([segment_length, ], tf.divide(1, noises[1])), sigmoid_vector)

    y = detrended_y_train

    ll_data_fit: tf.Tensor = tf.reduce_sum(y * tf.expand_dims(tf.add_n([diag_1, diag_2]), axis=-1) * y)

    ll_data_fit: tf.Tensor = -0.5 * tf.reshape(ll_data_fit, [])

    diag_1: tf.Tensor = tf.multiply(tf.fill([segment_length, ], noises[0]), _sigmoid_vector)
    diag_2: tf.Tensor = tf.multiply(tf.fill([segment_length, ], noises[1]), sigmoid_vector)
    log_determinant: tf.Tensor = tf.reduce_sum(tf.math.log(tf.add_n([diag_1, diag_2])))

    ll_complexity_penalty: tf.Tensor = tf.cast(-0.5, dtype=global_param.p_dtype) * log_determinant

    log_2_pi: tf.Tensor = tf.math.log(tf.math.multiply(tf.cast(np.pi, dtype=global_param.p_dtype),
                                                       tf.cast(2, dtype=global_param.p_dtype)))
    ll_norm_constant: tf.Tensor = tf.math.multiply(tf.cast(-0.5, dtype=global_param.p_dtype),
                                                   tf.math.multiply(tf.cast(
                                                       segment_length, global_param.p_dtype), log_2_pi))

    log_likelihood: tf.Tensor = tf.math.add_n([ll_data_fit, ll_complexity_penalty, ll_norm_constant])

    return -log_likelihood
