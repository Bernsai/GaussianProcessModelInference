import gpbasics.global_parameters as global_param

global_param.ensure_init()

import enum
import logging
import time
from typing import List, Tuple, Union
import tensorflow as tf
from gpmretrieval.ConcurrencyType import ConcurrencyType
import gpbasics.Metrics.Metrics as met
import gpbasics.DataHandling.DataInput as di
import gpbasics.MeanFunctionBasics.MeanFunction as mf
from gpmretrieval.KernelSearch import KernelSearch as ks
import gpmretrieval.KernelSearch.TopDownHierarchical as hks
import gpmretrieval.ChangePointDetection.SequentialCPDetection as scpd
import gpmretrieval.ChangePointDetection.BatchCPDetection as bcpd
import gpmretrieval.KernelSearch.ParallelApproach as pks
import gpmretrieval.KernelSearch.SequentialApproach as sks
import gpmretrieval.KernelSearch.IncrementalApproach as iks
import gpmretrieval.KernelSearch.TraversorType as trty
import gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy as kexp
import gpbasics.Statistics.GaussianProcess as gp
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpbasics.KernelBasics.Kernel import Kernel
from gpmretrieval.Partitioning.ClusteringMethodType import ClusteringMethod
import gpmretrieval.autogpmr_parameters as agpmr_param


class AlgorithmType(enum.Enum):
    # Compositional Kernel Search (CKS), cf. Duvenaud et al., Structure Discovery in Nonparametric Regression
    # through Compositional Kernel Search, at ICML 2013
    CKS = 0,

    # Automatic Bayesian Covariance Discovery (ABCD), cf. Lloyd et al., Automatic Construction and Natural-Language
    # Description of Nonparametric Regression Models, at AAAI 2014
    ABCD = 1,

    # Scalable Kernel Composition (SKC), cf. Kim and Teh, Scaling up the Automatic Statistician: Scalable Structure
    # Discovery using Gaussian Processes, at AISTATS 2018
    SKC = 2,

    # Called Concatenated Composite Covariance Search (3CS) in the literature, cf. Berns et al., 3CS Algorithm for
    # Efficient Gaussian Process Model Retrieval, at ICPR 2020
    SKS = 3,

    # Called Large-Scale Automatic Retrieval of Gaussian Process Models (LARGe) in the literature, cf. Berns and Beecks,
    # Automatic Gaussian Process Model Retrieval for Big Data, at CIKM 2020
    IKS = 4,

    # Called Lineage Gaussian Process Model Inference (LGI) in the literature, cf. Berns and Beecks, Complexity-Adaptive
    # Gaussian Process Model Inference for Large-Scale Data, at SDM 2021
    TopDown_HKS = 5


class GpmRetrieval:
    def __init__(self, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]], search_options: dict,
                 optimize_metric: met.MetricType, model_selection_metric: met.MetricType,
                 global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 covariance_function_repository: List[Kernel],
                 clustering_method: ClusteringMethod = None, subset_size: int = None,
                 sequential_cpd: scpd.SequentialChangePointDetectionType = None,
                 batch_cpd: bcpd.BatchChangePointDetectionType = None,
                 hierarchical_type: kexp.KernelExpansionStrategyType = None,
                 p_kernel_expression_replacement: bool = None, incremental_traversor: trty.TraversorType = None,
                 ignored_dimensions: List[int] = None, random_restart: int = 1):
        assert global_param.p_used_base_kernel, "Used based kernel are not initialized!"
        self.data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]] = data_input
        self.search_options: dict = search_options
        self.optimize_metric: met.MetricType = optimize_metric
        self.model_selection_metric: met.MetricType = model_selection_metric
        self.mean_function: mf.MeanFunction = None
        self.covariance_function_repository: List[Kernel] = covariance_function_repository

        if sequential_cpd is None:
            sequential_cpd = scpd.SequentialChangePointDetectionType.WhiteNoise
        self.sequential_cpd: scpd.SequentialChangePointDetectionType = sequential_cpd

        if batch_cpd is None:
            batch_cpd = bcpd.BatchChangePointDetectionType.BottomUp
        self.batch_cpd: bcpd.BatchChangePointDetectionType = batch_cpd
        self.global_approx: mht.GlobalApproximationsType = global_approx
        self.numerical_matrix_handling: mht.NumericalMatrixHandlingType = numerical_matrix_handling
        self.subset_size: int = subset_size

        if clustering_method is None:
            clustering_method = ClusteringMethod.KMEANS
        self.clustering_method: ClusteringMethod = clustering_method

        self.hierarchical_type = hierarchical_type

        if self.hierarchical_type is None:
            self.hierarchical_type = kexp.KernelExpansionStrategyType.SumOfProducts

        self.p_kernel_expression_replacement: bool = p_kernel_expression_replacement

        self.incremental_traversor: trty.TraversorType = incremental_traversor

        self.ignored_dimensions: List[int] = ignored_dimensions

        self.random_restarts: int = max(1, random_restart)

    def get_kernel_search_for_algorithm(self, alg_type: AlgorithmType, mean_function: mf.MeanFunction) \
            -> ks.KernelSearch:
        skc: bool = False

        global_approx: mht.GlobalApproximationsType = self.global_approx
        numerical_matrix_handling: mht.NumericalMatrixHandlingType = self.numerical_matrix_handling
        subset_size: int = self.subset_size

        if alg_type == AlgorithmType.CKS:
            assert 'global_max_depth' in self.search_options, "Missing search option 'global_max_depth' for CKS algorithm"
            search_type: ks.KernelSearchType = ks.KernelSearchType.Parallel
            expansion_type: kexp.KernelExpansionStrategyType = kexp.KernelExpansionStrategyType.CKS
        elif alg_type == AlgorithmType.ABCD:
            assert 'global_max_depth' in self.search_options, "Missing search option 'global_max_depth' for ABCD algorithm"
            search_type: ks.KernelSearchType = ks.KernelSearchType.Parallel
            expansion_type: kexp.KernelExpansionStrategyType = kexp.KernelExpansionStrategyType.ABCD
        elif alg_type == AlgorithmType.SKC:
            assert 'global_max_depth' in self.search_options, "Missing search option 'global_max_depth' for SKC algorithm"
            skc = True
            search_type: ks.KernelSearchType = ks.KernelSearchType.Parallel

            # The Paper "Scaling up the automatic statistician: Scalable structure discovery using gaussian processes"
            # by Kim and Teh (2018) leaves some ambiguity with regards to the kernel expansion mechanisms, since
            # CKS kernel expansion is defined there wrong, too. The authors do not explicate a reduction in candidate
            # kernel size in their text, but their pseudo code indicates so.
            # Therefore, these two options are given for SKC.
            if agpmr_param.p_skc_with_cks_strategy:
                expansion_type: kexp.KernelExpansionStrategyType = kexp.KernelExpansionStrategyType.CKS
            else:
                expansion_type: kexp.KernelExpansionStrategyType = kexp.KernelExpansionStrategyType.SKC
            global_approx = mht.MatrixApproximations.VFE
        elif alg_type == AlgorithmType.SKS:
            assert 'default_window_size' in self.search_options and 'npo' in self.search_options, \
                "Missing search option 'default_window_size' and/or 'npo' for SKS algorithm"
            search_type: ks.KernelSearchType = ks.KernelSearchType.SKS
            expansion_type: kexp.KernelExpansionStrategyType = self.hierarchical_type
        elif alg_type == AlgorithmType.IKS:
            assert 'npo' in self.search_options and 'local_max_depth' in self.search_options, \
                "Missing search option 'default_window_size' and/or 'max_npo' for IKS algorithm"
            search_type: ks.KernelSearchType = ks.KernelSearchType.Incremental
            expansion_type: kexp.KernelExpansionStrategyType = self.hierarchical_type
        elif alg_type == AlgorithmType.TopDown_HKS:
            assert 'npo' in self.search_options and 'local_max_depth' in self.search_options \
                   and 'partitions_split_per_layer' in self.search_options, \
                "Missing search option 'default_window_size' and/or 'max_npo' for TopDown_HKS algorithm"
            search_type: ks.KernelSearchType = ks.KernelSearchType.TopDown_HKS
            expansion_type: kexp.KernelExpansionStrategyType = self.hierarchical_type
        else:
            logging.critical("Invalid AlgorithmType given: '%s' " % alg_type)
            raise Exception("Invalid AlgorithmType given: '%s' " % alg_type)

        if search_type == ks.KernelSearchType.Parallel:
            if skc:
                if isinstance(self.data_input, list):
                    inducting_train = int(self.data_input[0].n_train * global_param.p_nystroem_ratio)
                else:
                    inducting_train = int(self.data_input.n_train * global_param.p_nystroem_ratio)
                logging.info("SKC will be executed with %i inducing data points." % inducting_train)
                kernel_search: ks.KernelSearch = pks.ParallelKernelSearch(
                    expansion_type, self.search_options, self.data_input, mean_function,
                    optimize_metric=met.MetricType.LL, model_selection_metric=met.MetricType.LL,
                    global_approx=global_approx, numerical_matrix_handling=numerical_matrix_handling,
                    subset_size=subset_size, p_kernel_expression_replacement=self.p_kernel_expression_replacement,
                    random_restart=self.random_restarts,
                    covariance_function_repository=self.covariance_function_repository, is_skc=True)

            else:
                kernel_search: ks.KernelSearch = pks.ParallelKernelSearch(
                    expansion_type, self.search_options, self.data_input, mean_function,
                    optimize_metric=self.optimize_metric, model_selection_metric=self.model_selection_metric,
                    global_approx=global_approx, numerical_matrix_handling=numerical_matrix_handling,
                    subset_size=subset_size, p_kernel_expression_replacement=self.p_kernel_expression_replacement,
                    random_restart=self.random_restarts,
                    covariance_function_repository=self.covariance_function_repository)

            kernel_search.activate_local_concurrency = True
            kernel_search.activate_global_multi_threading = False

        elif search_type == ks.KernelSearchType.SKS:
            kernel_search: ks.KernelSearch = sks.SequentialKernelSearch(
                self.data_input, mean_function, self.search_options["default_window_size"], self.search_options,
                expansion_type, sequential_cpd=self.sequential_cpd, optimize_metric=self.optimize_metric,
                model_selection_metric=self.model_selection_metric, global_approx=global_approx,
                numerical_matrix_handling=numerical_matrix_handling, subset_size=subset_size,
                p_kernel_expression_replacement=self.p_kernel_expression_replacement,
                random_restart=self.random_restarts, covariance_function_repository=self.covariance_function_repository)

        elif search_type == ks.KernelSearchType.Incremental:
            kernel_search: ks.KernelSearch = iks.IncrementalKernelSearch(
                expansion_type, self.search_options, self.data_input, self.mean_function,
                self.optimize_metric, self.model_selection_metric, cpd_type=self.batch_cpd,
                clustering_type=self.clustering_method, global_approx=global_approx,
                numerical_matrix_handling=numerical_matrix_handling,
                subset_size=subset_size, local_max_depth=self.search_options["local_max_depth"],
                p_kernel_expression_replacement=self.p_kernel_expression_replacement,
                incremental_traversor=self.incremental_traversor, ignored_dimensions=self.ignored_dimensions,
                random_restart=self.random_restarts, covariance_function_repository=self.covariance_function_repository)

        elif search_type == ks.KernelSearchType.TopDown_HKS:
            kernel_search: ks.KernelSearch = hks.TopDownHierarchicalApproach(
                expansion_type, self.search_options, self.data_input, self.mean_function, self.optimize_metric,
                self.model_selection_metric, cpd_type=self.batch_cpd, global_approx=global_approx,
                numerical_matrix_handling=numerical_matrix_handling, subset_size=subset_size,
                local_max_depth=self.search_options["local_max_depth"], clustering_method=self.clustering_method,
                p_kernel_expression_replacement=self.p_kernel_expression_replacement,
                ignored_dimensions=self.ignored_dimensions, random_restart=self.random_restarts,
                covariance_function_repository=self.covariance_function_repository)
        else:
            raise Exception("Invalid Kernel Search")

        return kernel_search

    def init_mean_function(self, mean_function: mf.MeanFunction = None):
        self.mean_function = mean_function

        if isinstance(self.data_input, list):
            [data_input_instance.set_mean_function(self.mean_function) for data_input_instance in self.data_input]
        else:
            self.data_input.set_mean_function(self.mean_function)

        mf_hyper_parameter_list: List[List[float]] = []

        logging.info(
            "Mean Function Set: %s: %s" % (self.mean_function.get_string_representation(), mf_hyper_parameter_list))

    def execute_kernel_search(self, algorithm_type: AlgorithmType) -> Tuple[gp.AbstractGaussianProcess, float]:
        assert self.mean_function is not None, \
            "Mean Function has not been initialized! Kernel Search cannot be performed."

        if algorithm_type is AlgorithmType.ABCD:
            global_param.p_cp_operator_type = global_param.ChangePointOperatorType.SIGMOID

        elif algorithm_type is AlgorithmType.SKC or self.global_approx is mht.MatrixApproximations.VFE or \
                self.global_approx is mht.MatrixApproximations.BASIC_NYSTROEM or \
                self.global_approx is mht.MatrixApproximations.FITC or \
                self.global_approx is mht.MatrixApproximations.SKI:
            actual_jitter: tf.constant = global_param.p_cov_matrix_jitter

            if algorithm_type is AlgorithmType.SKC:
                global_param.p_cov_matrix_jitter = tf.constant(1e-2, dtype=global_param.p_dtype)
            else:
                global_param.p_cov_matrix_jitter = tf.constant(1e-1, dtype=global_param.p_dtype)
            if actual_jitter != global_param.p_cov_matrix_jitter:
                logging.warning(
                    f"If Global Approx or SKC is used, we use a higher jitter of "
                    f"{global_param.p_cov_matrix_jitter} instead of {actual_jitter}.")

        elif algorithm_type is AlgorithmType.SKS or algorithm_type is AlgorithmType.IKS:
            global_param.p_cp_operator_type = global_param.ChangePointOperatorType.INDICATOR

        kernel_search: ks.KernelSearch = self.get_kernel_search_for_algorithm(
            algorithm_type, self.mean_function)

        logging.info("kernel search: optimize for %s, select models by %s"
                     % (kernel_search.optimize_metric, kernel_search.model_selection_metric))

        if global_param.p_max_threads > 1 and kernel_search.local_concurrency_type != ConcurrencyType.NONE:
            global_param.set_up_pool()

        start_seconds: float = time.time()
        best_gp: gp.AbstractGaussianProcess = kernel_search.start_kernel_search()
        end_seconds_search: float = time.time()
        elapsed_time_search: float = end_seconds_search - start_seconds

        logging.info("Kernel Search finished. Duration: %f s" % elapsed_time_search)

        best_gp.covariance_matrix.kernel.sort_child_nodes()
        logging.info("Result Kernel: %s" % (best_gp.kernel.get_string_representation()))

        return best_gp, elapsed_time_search
