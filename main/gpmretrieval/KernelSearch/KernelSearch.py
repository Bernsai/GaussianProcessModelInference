from gpbasics import global_parameters as global_param

global_param.ensure_init()

import threading
import logging
from functools import partial
from enum import Enum
from multiprocessing.managers import SyncManager
from gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy import KernelExpansionStrategy
from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput
from gpbasics.Metrics.MetricType import MetricType
from gpmretrieval.ConcurrencyType import ConcurrencyType
from typing import List, Tuple, Union
import gpbasics.Metrics.MatrixHandlingTypes as mht
import tensorflow as tf
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.Kernel as k
import gpbasics.MeanFunctionBasics.MeanFunction as mf
import gpbasics.Metrics.Metrics as met
import gpbasics.Optimizer.Fitter as f
import gpbasics.Statistics.GaussianProcess as gp
from gpbasics.DataHandling.BatchDataInput import BatchDataInput
import gpmretrieval.KernelExpansionStrategies.TreeStrategies as hkexp
from gpmretrieval.KernelExpansionStrategies import KernelExpansionStrategy as kexp
import gpmretrieval.KernelExpansionStrategies.CompositionalStrategies as ckexp
import gpbasics.Metrics.Auxiliary as met_aux
from gpmretrieval.autogpmr_parameters import p_model_selection_with_test_data


class KernelSearchType(Enum):
    Parallel = 0
    SKS = 2
    Incremental = 3
    TopDown_HKS = 4


class KernelSearch:
    def get_strategy(self, options: dict) -> KernelExpansionStrategy:
        """
        This method returns a KernelExpansionStrategy object initialized with the provided options and complying with
        the KernelExpansionStrategyType specified as part of this KernelSearch object.
        Args:
            options: dictionary of parameters for KernelExpansionStrategy

        Returns: KernelExpansionStrategy

        """
        return self.get_strategy_by_type(self.strategy_type, options)

    def get_strategy_by_type(self, strategy_type: kexp.KernelExpansionStrategyType, options: dict) \
            -> KernelExpansionStrategy:
        """
        This method returns a KernelExpansionStrategy object initialized with the provided options and complying with
        the given KernelExpansionStrategyType.
        Args:
            strategy_type: KernelExpansionStrategyType
            options: dictionary of parameters for KernelExpansionStrategy

        Returns: KernelExpansionStrategy

        """

        # if provided data input is a list of several data input (as required for k-fold cross validation), the input
        # dimensionality of the first data input of that list is taken as representative
        # for all data input in that list.
        if isinstance(self.data_input, list):
            dim = self.data_input[0].get_input_dimensionality()
        else:
            dim = self.data_input.get_input_dimensionality()

        def switch_strategy_cks(cks_option: dict, p_kernel_expression_replacement):
            strategy = ckexp.CksStrategy(
                max_depth=cks_option["global_max_depth"], input_dimensionality=dim,
                kernel_expression_replacement=p_kernel_expression_replacement,
                covariance_function_repository=self.covariance_function_repository)
            return strategy

        def switch_strategy_skc(skc_option: dict, p_kernel_expression_replacement):
            strategy = ckexp.SkcStrategy(
                max_depth=skc_option["global_max_depth"], input_dimensionality=dim,
                covariance_function_repository=self.covariance_function_repository)
            return strategy

        def switch_strategy_abcd(abcd_options: dict, p_kernel_expression_replacement):
            strategy = ckexp.AbcdStrategy(
                max_depth=abcd_options["global_max_depth"], input_dimensionality=dim,
                kernel_expression_replacement=p_kernel_expression_replacement,
                covariance_function_repository=self.covariance_function_repository)
            return strategy

        def switch_strategy_basic_hierarchical(basic_hierarchical_options: dict, p_kernel_expression_replacement):
            if "local_max_depth" in basic_hierarchical_options:
                max_depth = basic_hierarchical_options["local_max_depth"]
            else:
                logging.info(
                    "No local max depth given for basic hierarchical kernel expansion strategy. Set to 'None'.")
                max_depth = None

            if "npo" in basic_hierarchical_options:
                max_npo = basic_hierarchical_options["npo"]
            else:
                logging.info(
                    "No max npo (Nodes Per Operator) given for basic hierarchical "
                    "kernel expansion strategy. Set to 'None'.")
                max_npo = None

            strategy = hkexp.SumOfProductsExpansionStrategy(
                max_npo=max_npo, max_depth=max_depth, input_dimensionality=dim,
                kernel_expression_replacement=p_kernel_expression_replacement,
                covariance_function_repository=self.covariance_function_repository)
            return strategy

        switch_strategy = {
            kexp.KernelExpansionStrategyType.SumOfProducts: switch_strategy_basic_hierarchical,
            kexp.KernelExpansionStrategyType.CKS: switch_strategy_cks,
            kexp.KernelExpansionStrategyType.ABCD: switch_strategy_abcd,
            kexp.KernelExpansionStrategyType.SKC: switch_strategy_skc
        }

        return switch_strategy.get(strategy_type)(options, self.p_kernel_expression_replacement)

    def __init__(
            self, strategy_type: kexp.KernelExpansionStrategyType,
            data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]], mean_function: mf.MeanFunction,
            kernel_search_type: KernelSearchType, optimize_metric: MetricType,
            model_selection_metric: MetricType, p_kernel_expression_replacement: bool,
            global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
            covariance_function_repository: List[k.Kernel], random_restart: int = 1, subset_size: int = None):
        self.kernel_search_type: KernelSearchType = kernel_search_type
        self.strategy_type: kexp.KernelExpansionStrategyType = strategy_type
        self.data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]] = data_input
        self.optimize_metric: MetricType = optimize_metric
        self.model_selection_metric: MetricType = model_selection_metric
        self.mean_function: mf.MeanFunction = mean_function

        # Global concurrency is used, when multiple independent kernel searches are employed. This applies to all the
        # divisive kernel search approaches like SKS, IKS, (TopDown_HKS) (or in the literature 3CS, LARGe, (LGI)). CKS,
        # ABCD, SKC on the other hand do not use this kind of concurrency.
        self.activate_global_multi_threading: bool = None

        # Local concurrency is used, when multiple candidate kernel expressions are evaluated simultaneously. This can
        # be especially used for CKS, ABCD, and SKC.
        self.local_concurrency_type: ConcurrencyType = None

        self.strategy_options: dict = {}
        self.global_approx: mht.GlobalApproximationsType = global_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size
        self.p_kernel_expression_replacement: bool = p_kernel_expression_replacement
        self.random_restarts: int = max(1, random_restart)
        if covariance_function_repository is None:
            self.covariance_function_repository: List[k.Kernel] = []
        else:
            self.covariance_function_repository: List[k.Kernel] = covariance_function_repository

        self.use_covfunc_repo: bool = len(self.covariance_function_repository) > 0
        self.is_skc: bool = False

    def start_kernel_search(self) -> gp.AbstractGaussianProcess:
        """
        Calling this function means to execute the actual kernel search.
        Returns: the found optimal Gaussian process model

        """
        pass

    def single_kernel_optimization(
            self, kernel: k.Kernel, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]],
            differing_optimize_metric: MetricType = None) -> Tuple[gp.GaussianProcess, tf.Tensor]:
        """
        This method encapsulates all steps for optimizing a Gaussian process initialized with the given kernel /
        covariance function for the specified data.

        Args:
            kernel: the kernel / covariance function, whose hyper parameters should be optimized
            data_input: the specified data
            differing_optimize_metric: if an optimize_metric, which differs from the one specified with the kernel
            search object, should be used, it can be specified via this parameter

        Returns: A Tuple, which holds the optimized Gaussian process (including the respective covariance function), and
        the model selection measure evaluated via that Gaussian process and the specified data.

        """
        inducing_points = None
        try:
            with tf.name_scope("gp"):
                _gp: gp.GaussianProcess = gp.GaussianProcess(kernel, self.mean_function)

            # Since indicator change points (cf. SKS / 3CS) are not optimized via "single_kernel_optimization"
            # but a separate change point detection procedure, one can assume that calling "single_kernel_optimization"
            # with a ChangePointOperator-Kernel originated in executing ABCD.
            if self.strategy_type == kexp.KernelExpansionStrategyType.ABCD:
                global_param.p_cp_operator_type = global_param.ChangePointOperatorType.SIGMOID

            # If a repository of covariance function is used, these are not optimized, since their hyperparameters are
            # regarded as unchangeable. Therefore, the selective metric has to be determined separately in
            # in this case, no matter what.
            if not self.use_covfunc_repo:
                with tf.name_scope("fitter"):
                    if differing_optimize_metric is not None:
                        om: MetricType = differing_optimize_metric
                    else:
                        om: MetricType = self.optimize_metric

                    fitter: f.Fitter = \
                        global_param.p_fitter(
                            data_input, _gp, om, global_approx=self.global_approx,
                            from_distribution=self.random_restarts > 1,
                            numerical_matrix_handling=self.numerical_matrix_handling, subset_size=self.subset_size)

                metric_results: List[Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor], tf.Tensor]] = []

                for i in range(self.random_restarts):
                    try:
                        metric_results.append(fitter.fit())
                    except Exception as e:
                        print(e)

                _, resulted_metric, hyper_parameter, inducing_points = \
                    sorted(metric_results, key=lambda single_result: single_result[1])[0]

                _gp.covariance_matrix.kernel.set_hyper_parameter(hyper_parameter)
                _gp.set_inducing_points(inducing_points)

            resulted_gp = _gp

            if p_model_selection_with_test_data or not self.optimize_metric == self.model_selection_metric or \
                    self.is_skc or self.use_covfunc_repo:

                # SKC uses VFE as an optimization metric and its own specific upper bound for model selection
                if self.is_skc:
                    global_approx = mht.MatrixApproximations.SKC_UPPER_BOUND
                else:
                    global_approx = self.global_approx

                selective_metrics = [self.get_metric(_gp, entry[2], global_approx, self.data_input, inducing_points)
                                     for entry in metric_results]

                selective_metric, metric_result = \
                    sorted(zip(selective_metrics, metric_results), key=lambda single_result: single_result[0])[0]

                _, _, hyper_parameter, inducing_points = metric_result

                _gp.covariance_matrix.kernel.set_hyper_parameter(hyper_parameter)
                _gp.set_inducing_points(inducing_points)

                resulted_metric = tf.reduce_min(selective_metrics)
                
        except (tf.errors.InvalidArgumentError, IndexError) as e:
            logging.error(
                "KernelSearch.single_kernel_optimization: k" + kernel.get_string_representation() + ":" + str(e))
            return None, 1e+399

        resulted_gp.covariance_matrix.reset()
        resulted_gp.aux.reset()
        return resulted_gp, resulted_metric

    def single_kernel_optimization_parallelizable(
            self, kernel: k.Kernel, lock_results, result_gps: List[gp.GaussianProcess], result_metrics: List[float],
            data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]]):
        """
        This method wraps the functionality of the method 'single_kernel_optimization' in a way, that it supports
        mechanisms for executing multiple instances of it concurrently. Therefore, result lists and a respectice
        concurrency lock for them is provided, which are synced across processes and threads.
        Args:
            kernel: the kernel / covariance function, whose hyper parameters should be optimized
            lock_results: lock, which needs to be acquired before changing the following two result lists
            result_gps: list of all concurrently evaluated Gaussian process models
            result_metrics: list of the selective quality measures
            for all concurrently evaluated Gaussian process models
            data_input: the specified data

        Returns:

        """

        outcome: Tuple[gp.GaussianProcess, tf.Tensor] = \
            self.single_kernel_optimization(kernel, data_input)

        if outcome[0] is not None and outcome[1] is not None:
            outcome_gp: gp.GaussianProcess = outcome[0]
            outcome_metric: tf.Tensor = outcome[1]

            lock_results.acquire()
            result_metrics.append(outcome_metric)
            lock_results.release()

            lock_results.acquire()
            result_gps.append(outcome_gp)
            lock_results.release()

    def get_metric(self, gaussian_process: gp.AbstractGaussianProcess, hyper_parameter: List[tf.Tensor],
                   global_approx: mht.GlobalApproximationsType,
                   data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]], inducing_points) \
            -> tf.Tensor:
        """
        This auxiliary method helps to calculate the model selection metric for a Gaussian process model and accounts
        for various circumstances.
        Args:
            gaussian_process: the Gaussian process model (GPM)
            hyper_parameter: the corresponding hyper parameter of the GPM's covariance function
            global_approx: the used global approximation
            data_input:  the respective data
            inducing_points: inducing points, if applicable

        Returns: the selective measure

        """
        if isinstance(self.data_input, list):
            selective_metrics = [
                self.get_metric(gaussian_process, hyper_parameter, global_approx, data_input_instance, inducing_points)
                for data_input_instance in self.data_input]

            return tf.reduce_mean(selective_metrics)
        else:
            if p_model_selection_with_test_data and self.model_selection_metric != MetricType.MSE:
                data_input = get_flipped_data_input(data_input)

            gaussian_process.set_data_input(data_input)
            selective_metric: met.AbstractMetric = met_aux.get_metric_by_type(
                self.model_selection_metric, gaussian_process, global_approx, self.numerical_matrix_handling,
                self.subset_size)
            selective_metric: tf.Tensor = selective_metric.get_metric(hyper_parameter, inducing_points)
            return selective_metric

    def perform_kernel_search_internal(
            self, strategy: Union[kexp.KernelExpansionStrategy, dict], local_concurrency_type: ConcurrencyType,
            data_input: Union[AbstractDataInput, List[AbstractDataInput]]) -> gp.AbstractGaussianProcess:
        """
        This method encapsulates a full kernel search with regards to specified data.
        Args:
            strategy: this parameter either specifies parameters for a KernelExpansionStrategy-object or the
            KernelExpansionStrategy-object itself.
            local_concurrency_type: the kind of local concurrency, which ought to be applied, if possible.
            data_input: the respective data

        Returns: a Gaussian process model optimal with regards to the specified optimization and model selection
        measures.

        """
        if not isinstance(strategy, kexp.KernelExpansionStrategy):
            strategy: kexp.KernelExpansionStrategy = self.get_strategy(strategy)

        # local concurrency can only be used, if global concurrency is not used, since parallelizing both ways is not
        # possible with python multiprocessing and multithreading
        concurrency: bool = \
            global_param.p_max_threads > 1 and local_concurrency_type != ConcurrencyType.NONE and \
            not self.activate_global_multi_threading

        # if local concurrency is used, synchronization mechanisms need to be used. These are provided by SyncManager.
        if concurrency:
            manager: SyncManager = SyncManager()
            manager.start()
        else:
            manager = None

        gaussian_process: gp.AbstractGaussianProcess = None
        gp_metric: float = None
        further_expansion_possible: bool = True

        while further_expansion_possible:
            if gaussian_process is None:
                logging.info("Kernel Search: Evaluating Base Kernels.")
            else:
                logging.info(f"Kernel Search: Evaluating Composite Kernels at depth "
                             f"{gaussian_process.kernel.get_number_base_kernels() + 1}.")
            gaussian_process, gp_metric, further_expansion_possible = self.perform_kernel_search_iteration(
                gaussian_process, gp_metric, data_input, strategy, local_concurrency_type, manager)

        logging.info(f"ps best: {gaussian_process.covariance_matrix.kernel.get_string_representation()}: "
                     f"{gp_metric}")

        global_param.shutdown_pool()

        if concurrency:
            manager.shutdown()

        return gaussian_process

    def perform_kernel_search_iteration(
            self, previous_gp: gp.AbstractGaussianProcess, previous_metric: float,
            data_input: Union[AbstractDataInput, List[AbstractDataInput]], strategy: kexp.KernelExpansionStrategy,
            local_concurrency_type: ConcurrencyType, manager: SyncManager) \
            -> Tuple[gp.AbstractGaussianProcess, float, bool]:
        """
        This method performs one iteration of a kernel search procedure.
        Args:
            previous_gp: If this is not the first iteration, this iteration generates candidates based on the previous
            best candidate, which is embodied by this parameter.
            previous_metric: quality measure of the previous best candidate
            data_input: data input used for kernel search
            strategy: applied strategy for candidate generation
            local_concurrency_type: whether to evaluate multiple candidate kernels / candidate gaussian process
            models at once and how (multi threading or multi processing).
            manager: needed for multi processing

        Returns: a tuple of three elements. First element gives the best candidate (possibly the previous best). Second
        element describes the quality of that candidate, and third element indicates, whether this new best candidate is
        really new (True) or the best candidate is still the previous candidate (False)

        """

        concurrency: bool = global_param.p_max_threads > 1 and local_concurrency_type != ConcurrencyType.NONE and \
                            not self.activate_global_multi_threading

        if previous_gp is None:
            origin_kernels = strategy.get_initial_kernels()
            previous_metric = 1e+399
        else:
            origin_kernels, only_replacements = strategy.expand_kernel(previous_gp.kernel, data_input=data_input)

        # No further expansion possible
        if len(origin_kernels) == 0:
            if previous_gp is None:
                raise Exception("No kernel expansion possible in first iteration.")

            logging.warning(
                "Started 'perform_kernel_search_iteration' without the possibility of further kernel expansion.")
            return previous_gp, previous_metric, False

        self_setup_manager = False

        if concurrency:
            if manager is None:
                manager: SyncManager = SyncManager()
                manager.start()
                self_setup_manager = True

            lock_results = manager.Lock()
            result_metrics = manager.list([])
            result_gps = manager.list([])

            if local_concurrency_type == ConcurrencyType.MULTI_PROCESSING:
                iterable_arg_tuples = []

                for kernel in origin_kernels:
                    iterable_arg_tuples.append((kernel, lock_results, result_gps, result_metrics, data_input))

                partial_wrapped = partial(
                    self.single_kernel_optimization_parallelizable, lock_results=lock_results, result_gps=result_gps,
                    result_metrics=result_metrics, data_input=data_input)

                d = global_param.pool.map(partial_wrapped, origin_kernels)
            elif local_concurrency_type == ConcurrencyType.MULTI_THREADING:
                threads_list: List[threading.Thread] = []
                for kernel in origin_kernels:
                    while threading.active_count() > global_param.p_max_threads:
                        one_t: threading.Thread = threads_list.pop(0)
                        one_t.join()

                    t: threading.Thread = \
                        threading.Thread(target=self.single_kernel_optimization_parallelizable,
                                         args=(kernel, lock_results, result_gps, result_metrics, strategy,
                                               data_input, False))
                    t.start()
                    threads_list.append(t)

                for t in threads_list:
                    t.join()
            else:
                raise Exception(f"Invalid ConcurrencyType '{local_concurrency_type}'.")
        else:
            result_metrics = []
            result_gps = []
            for kernel in origin_kernels:
                outcome: Tuple[gp.GaussianProcess, tf.Tensor] = self.single_kernel_optimization(kernel, data_input)

                if outcome is not None and len(outcome) > 1 and outcome[0] is not None and outcome[1] is not None:
                    result_gps.append(outcome[0])
                    result_metrics.append(outcome[1])

        if len(result_metrics) == 0 or len(result_gps) == 0:
            return previous_gp, previous_metric, False

        sorted_results: List[Tuple[gp.AbstractGaussianProcess, tf.Tensor]] = \
            [(_gp, metric) for metric, _gp in
             sorted(zip(result_metrics, result_gps), key=lambda result_gaussian_process: result_gaussian_process[0])]

        if self_setup_manager:
            manager.shutdown()

        best_gp: gp.AbstractGaussianProcess = sorted_results[0][0]

        best_metric: float = float(sorted_results[0][1])

        if best_metric < previous_metric:
            return best_gp, best_metric, strategy.further_expansion_possible(best_gp.kernel)
        else:
            return previous_gp, previous_metric, False


def get_flipped_data_input(data_input: AbstractDataInput):
    """
    This is an auxiliary function, which swaps training and test data.
    Args:
        data_input: the respective data input

    Returns: flipped data input

    """
    if isinstance(data_input, BatchDataInput):
        flipped_data_input = BatchDataInput(
            data_input.data_x_test, data_input.data_y_test, data_input.data_x_train, data_input.data_y_train)
    else:
        flipped_data_input = di.DataInput(
            data_input.data_x_test, data_input.data_y_test, data_input.data_x_train, data_input.data_y_train)

    flipped_data_input.set_mean_function(data_input.mean_function)
    return flipped_data_input