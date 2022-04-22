from gpbasics import global_parameters as global_param

global_param.ensure_init()

import logging
from typing import List, Union
from gpmretrieval.ConcurrencyType import ConcurrencyType
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.Kernel as k
import gpbasics.Metrics.MatrixHandlingTypes as mht
import gpbasics.Statistics.GaussianProcess as gp
from gpmretrieval.KernelExpansionStrategies import KernelExpansionStrategy as kexp
from gpbasics.MeanFunctionBasics import MeanFunction as mf
from gpbasics.Metrics import Metrics as met
from gpmretrieval.KernelSearch.KernelSearch import KernelSearch, KernelSearchType


class ParallelKernelSearch(KernelSearch):
    def __init__(self, strategy_type: kexp.KernelExpansionStrategyType, strategy_options: dict,
                 data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]],
                 mean_function: mf.MeanFunction, optimize_metric: met.MetricType,
                 model_selection_metric: met.MetricType, p_kernel_expression_replacement: bool,
                 global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 covariance_function_repository: List[k.Kernel], random_restart: int = 1, subset_size: int = None,
                 is_skc: bool = False):
        super(ParallelKernelSearch, self).__init__(
            strategy_type, data_input, mean_function, KernelSearchType.Parallel, optimize_metric,
            model_selection_metric, p_kernel_expression_replacement, global_approx, numerical_matrix_handling,
            covariance_function_repository, random_restart, subset_size)

        self.strategy_options: dict = strategy_options
        self.default_logging = logging.info

        # ParallelKernelSearch methods (CKS, ABCD, SKC) do not have an overarching divisive approach (cf. partitioning),
        # thus a global concurrency approach is not applicable here, in that sense.
        self.activate_global_multi_threading: bool = False
        self.local_concurrency_type = ConcurrencyType.MULTI_PROCESSING
        self.is_skc = is_skc

    def start_kernel_search(self) -> gp.AbstractGaussianProcess:
        return self.perform_kernel_search()

    def perform_kernel_search(self) -> gp.AbstractGaussianProcess:
        return self.perform_kernel_search_internal(
            self.strategy_options, self.local_concurrency_type, self.data_input)
