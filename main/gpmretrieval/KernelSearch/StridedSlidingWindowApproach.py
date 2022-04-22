import gpbasics.global_parameters as global_param

global_param.ensure_init()

import logging
from gpmretrieval.ConcurrencyType import ConcurrencyType
from typing import List, Union
import tensorflow as tf
import gpbasics.DataHandling.DataInput as di
import gpbasics.DataHandling.BatchDataInput as bdi
import gpbasics.KernelBasics.Kernel as k
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.Statistics.GaussianProcess as gp
from gpmretrieval.KernelExpansionStrategies import KernelExpansionStrategy as kexp
from gpbasics import global_parameters as global_param
from gpbasics.MeanFunctionBasics import MeanFunction as mf
from gpbasics.Metrics import Metrics as met
from gpmretrieval.KernelSearch.KernelSearch import KernelSearch, KernelSearchType


def min_max_normalize(data):
    data = data - tf.reduce_min(data, axis=-2, keepdims=True)
    data = data / tf.reduce_max(data, axis=-2, keepdims=True)
    return data


class StridedKernelSearch(KernelSearch):
    def __init__(self, strategy_type: kexp.KernelExpansionStrategyType, strategy_options: dict,
                 data_input: Union[di.DataInput, bdi.BatchDataInput], mean_function: mf.MeanFunction,
                 optimize_metric: met.MetricType,
                 model_selection_metric: met.MetricType, p_kernel_expression_replacement: bool,
                 global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 normalize_per_window: bool, window_size: int, stride: int,
                 covariance_function_repository: List[k.Kernel] = None, random_restart: int = 1,
                 subset_size: int = None, pool_max_tasks_per_child: int = -1):
        super(StridedKernelSearch, self).__init__(
            strategy_type, data_input, mean_function, KernelSearchType.Parallel, optimize_metric,
            model_selection_metric, p_kernel_expression_replacement, global_approx, numerical_matrix_handling,
            covariance_function_repository, random_restart, subset_size)
        self.strategy_options: dict = strategy_options

        self.local_max_depth: int
        if "local_max_depth" in strategy_options:
            self.local_max_depth = int(strategy_options["local_max_depth"])
        else:
            self.local_max_depth = 4

        self.default_logging = logging.info
        self.window_size: int = window_size
        self.stride: int = stride
        self.normalize_per_window = normalize_per_window
        self.local_concurrency_type = ConcurrencyType.MULTI_THREADING
        self.activate_global_multi_threading: bool = True

        if pool_max_tasks_per_child is None or pool_max_tasks_per_child < 1 or not isinstance(pool_max_tasks_per_child, int):
            self.pool_max_tasks_per_child = None
        else:
            self.pool_max_tasks_per_child = pool_max_tasks_per_child

    def get_sliding_windowed_data(self, data_input: di.DataInput):
        if isinstance(data_input, bdi.BatchDataInput):
            x_train = tf.signal.frame(data_input.data_x_train, self.window_size, self.stride, axis=-2, pad_end=True)

            y_train = tf.signal.frame(data_input.data_y_train, self.window_size, self.stride, axis=-2, pad_end=True)

            x_test = tf.signal.frame(data_input.data_x_test, self.window_size, self.stride, axis=-2, pad_end=True)

            y_test = tf.signal.frame(data_input.data_y_test, self.window_size, self.stride, axis=-2, pad_end=True)

            if not data_input.same_index_values:
                x_train = tf.transpose(x_train, perm=[1, 0, 2, 3])
                x_test = tf.transpose(x_test, perm=[1, 0, 2, 3])

            y_train = tf.transpose(y_train, perm=[1, 0, 2, 3])
            y_test = tf.transpose(y_test, perm=[1, 0, 2, 3])
        else:
            x_train = tf.signal.frame(data_input.data_x_train, self.window_size, self.stride, axis=-1, pad_end=True)

            y_train = tf.repeat(tf.expand_dims(data_input.data_y_train, axis=0), axis=0, repeats=x_train.shape[1])

            x_test = tf.signal.frame(data_input.data_x_test, self.window_size, self.stride, axis=-1, pad_end=True)

            y_test = tf.repeat(tf.expand_dims(data_input.data_y_test, axis=0), axis=0, repeats=x_test.shape[1])

            x_train = tf.transpose(x_train, perm=[1, 0, 2])
            x_test = tf.transpose(x_test, perm=[1, 0, 2])

        return x_train, y_train, x_test, y_test

    def start_kernel_search(self) -> List[gp.AbstractGaussianProcess]:
        sliding_windowed_data_ = zip(*list(self.get_sliding_windowed_data(self.data_input)))

        agg_str = str(global_param.p_batch_metric_aggregator).split(" ")[1]

        sliding_windowed_data = [entry_tuple + (agg_str,) for entry_tuple in sliding_windowed_data_]
        del sliding_windowed_data_

        if global_param.p_max_threads > 1:
            global_param.set_up_pool(self.pool_max_tasks_per_child)
            result = global_param.pool.map(self.get_model, sliding_windowed_data)
            global_param.shutdown_pool()
        else:
            result = list(map(self.get_model, sliding_windowed_data))

        result = list(filter(None, result))

        return result

    def get_model(self, args):
        data_input = self.get_data_inputs(args[:-1])

        agg_str = args[-1:][0]

        if "min" in agg_str:
            global_param.p_batch_metric_aggregator = tf.math.reduce_min
        elif "max" in agg_str:
            global_param.p_batch_metric_aggregator = tf.math.reduce_max
        elif "mean" in agg_str:
            global_param.p_batch_metric_aggregator = tf.math.reduce_mean
        elif "std" in agg_str:
            global_param.p_batch_metric_aggregator = tf.math.reduce_std
        else:
            raise Exception("Invalid Metric Aggregator")

        result_gpm = self.perform_kernel_search_internal(
            self.strategy_options, self.local_concurrency_type, data_input)

        del result_gpm.data_input

        del result_gpm.covariance_matrix.data_input

        del result_gpm.aux.data_input

        result_gpm.covariance_matrix.reset()

        del args

        del data_input

        return result_gpm

    def get_data_inputs(self, args) -> di.DataInput:
        assert len(args) % 4 == 0

        data_input = self.get_data_input(*(args[0:4]))

        return data_input

    def get_data_input(self, data_x_train, data_y_train, data_x_test, data_y_test):
        if self.normalize_per_window:
            data_x_train = min_max_normalize(data_x_train)
            data_x_test = min_max_normalize(data_x_test)

        if len(data_y_train.shape) == 3:
            data_input = bdi.BatchDataInput(data_x_train, data_y_train, data_x_test, data_y_test, sort=False)
        elif len(data_y_train.shape) == 2:
            data_input = di.DataInput(data_x_train, data_y_train, data_x_test, data_y_test, sort=False)
        else:
            raise Exception("Invalid data input")

        data_input.set_mean_function(self.mean_function)

        return data_input
