import multiprocessing
import os
import sys
import logging

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

# This file has to be put directly in folder "main/"
#   all other dependent packages need to be in sub folders of "main/"
if os.name != 'nt':
    sys.path.append(os.getcwd())

import gpbasics.global_parameters as global_param
global_param.init(tf_parallel=8)

import numpy as np
import tensorflow as tf
import gpbasics.KernelBasics.BaseKernels as bk
from gpbasics.DataHandling.DataInput import DataInput
import gpbasics.Metrics.Metrics as met
from gpbasics.MeanFunctionBasics.BaseMeanFunctions import ZeroMeanFunction
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpbasics.Metrics.Auxiliary import get_metric_by_type
import gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy as kexp
import gpmretrieval.autogpmr_parameters as auto_gpm_param
from gpmretrieval.AutomaticGpmRetrieval import GpmRetrieval, AlgorithmType

global_param.p_max_threads = 1
global_param.p_used_base_kernel = [
    bk.PeriodicKernel, bk.SquaredExponentialKernel, bk.LinearKernel, bk.MaternKernel3_2, bk.MaternKernel5_2]

global_param.p_default_hierarchical_kernel_expansion = \
    kexp.KernelExpansionStrategyType.SumOfProducts

auto_gpm_param.p_model_selection_with_test_data = False
auto_gpm_param.p_optimizable_noise = False

global_param.p_dtype = tf.float64

global_param.p_cov_matrix_jitter = tf.constant(1e-6, dtype=global_param.p_dtype)

global_param.p_check_hyper_parameters = False

global_param.p_nystroem_ratio = 0.1

global_param.p_logging_level = logging.DEBUG

if __name__ == '__main__':
    search_options = {
      "local_max_depth": 4,
      "global_max_depth": 4,
      "npo": 5,
    }

    n = 500
    x = tf.reshape(tf.cast(tf.linspace(0, 1, n), dtype=global_param.p_dtype), (-1, 1))
    y = tf.math.sin(x * np.pi) + tf.cast(tf.random.normal(x.shape, mean=0.0, stddev=0.1), dtype=global_param.p_dtype)

    data_input = DataInput(x, y, test_ratio=0.1, seed=5499274)

    gpmi = GpmRetrieval(
        data_input, search_options, met.MetricType.LL, met.MetricType.LL, mht.MatrixApproximations.NONE,
        mht.NumericalMatrixHandlingType.CHOLESKY_BASED, None, random_restart=1)

    gpmi.init_mean_function(ZeroMeanFunction(data_input.get_input_dimensionality()))

    found_gpm, gpmi_runtime = gpmi.execute_kernel_search(AlgorithmType.IKS)

    rmse = float(get_metric_by_type(
        met.MetricType.RMSE, found_gpm).get_metric(found_gpm.kernel.get_hyper_parameter()).numpy())

    logging.info(f"Inferred GPM within {gpmi_runtime} seconds delivering RMSE of {rmse}")

