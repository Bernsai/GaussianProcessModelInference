from gpbasics import global_parameters as global_param

global_param.ensure_init()

from typing import List, Type, Tuple, cast, Union
from gpbasics.KernelBasics.Kernel import Kernel
import tensorflow as tf
import gpbasics.DataHandling.DataInput as di
from gpbasics.KernelBasics import Operators as op, Kernel as k
from gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy import KernelExpansionStrategyType, KernelExpansionStrategy
import gpmretrieval.autogpmr_parameters as autogpmi_param
from gpbasics.KernelBasics.BaseKernels import WhiteNoiseKernel
from gpbasics.KernelBasics.Operators import AdditionOperator, MultiplicationOperator


class CksStrategy(KernelExpansionStrategy):
    """
    The CKS Kernel Expansion Strategy is the one proposed by Duvenaud et al. (Structure discovery in nonparametric
    regression through compositional kernel search, at ICML 2013). The basic idea is to generate candidate kernel
    expressions by extending upon the current best performing covariance function / kernel. In doing so, a candidates
    are generated by considering an expansion of every possible subexpressions by means of an operator (Addition,
    and Multiplication) and a base kernel.
    """
    def __init__(self, max_depth: int, input_dimensionality: int, kernel_expression_replacement: bool,
                 covariance_function_repository: List[k.Kernel]):
        super(CksStrategy, self).__init__(
            KernelExpansionStrategyType.CKS, input_dimensionality, kernel_expression_replacement,
            covariance_function_repository)

        self.operators: \
            List[Union[Type[op.AdditionOperator], Type[op.MultiplicationOperator], Type[op.ChangePointOperator]]] \
            = [op.MultiplicationOperator, op.AdditionOperator]

        # maximum amount of constituent base kernels in a kernel
        self.max_depth: int = max_depth

        # Since the kernel replacement tactic is standard for CKS and ABCD (and arguably SKC), no setting of that
        # parameter is assumed to mean its activation
        self.kernel_expression_replacement = \
            self.kernel_expression_replacement or self.kernel_expression_replacement is None

    def get_initial_kernels(self) -> List[k.Kernel]:
        origin_kernels: List[k.Kernel] = []
        for i in range(0, self.get_number_of_base_kernels()):
            origin_kernels.append(self.get_base_kernel_by_index(i))

        dim = self.input_dimensionality
        if autogpmi_param.p_optimizable_noise:
            return [op.AdditionOperator(dim, [WhiteNoiseKernel(dim), origin_kernel])
                    for origin_kernel in origin_kernels]
        else:
            return origin_kernels

    def expand_kernel(self, previous_kernel: Kernel, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]]) \
            -> Tuple[List[k.Kernel], bool]:
        if autogpmi_param.p_optimizable_noise:
            previous_kernel = previous_kernel.child_nodes[1]

        if isinstance(data_input, list):
            xrange = data_input[0].get_x_range()
        else:
            xrange = data_input.get_x_range()

        new_kernel_candidates = []
        only_replacements = False
        dim = self.input_dimensionality

        paths = self.get_paths_to_all_subexpressions(previous_kernel)
        for i in range(0, self.get_number_of_base_kernels()):
            if previous_kernel.get_number_base_kernels() < self.max_depth:

                # 1. Candidate Generation by Kernel Expansions on lower levels of the concatenated kernel expression
                for expand_op in self.operators:
                    for p in paths:
                        copied_previous_kernel = previous_kernel.deepcopy()
                        current_node = copied_previous_kernel
                        for j in range(0, len(p) - 1):
                            current_node = current_node.child_nodes[p[j]]
                        expand_bk = self.get_base_kernel_by_index(i)
                        if expand_op == op.ChangePointOperator:
                            new_cp: tf.Tensor = \
                                cast(tf.Tensor, tf.constant([(x[0] + x[1]) / 2 for x in xrange], shape=[dim, ],
                                                            dtype=global_param.p_dtype))

                            new_child_node = op.ChangePointOperator(
                                dim, [current_node.child_nodes[p[len(p) - 1]], expand_bk], [new_cp])
                        else:
                            new_child_node = \
                                expand_op(dim, [current_node.child_nodes[p[len(p) - 1]], expand_bk])

                        if self.kernel_expression_replacement is None \
                                or self.kernel_expression_replacement:
                            current_node.replace_child_node(p[len(p) - 1], new_child_node)

                        new_kernel_candidates.append(copied_previous_kernel)

                # 2. Candidate Generation by Kernel Expansions on highest level of the concatenated kernel expression
                copied_previous_kernel = previous_kernel.deepcopy()
                new_kernel_candidates.append(op.AdditionOperator(
                    self.input_dimensionality, [copied_previous_kernel, self.get_base_kernel_by_index(i)]))

                copied_previous_kernel = previous_kernel.deepcopy()
                new_kernel_candidates.append(op.MultiplicationOperator(
                    self.input_dimensionality, [copied_previous_kernel, self.get_base_kernel_by_index(i)]))

                if op.ChangePointOperator in self.operators:
                    copied_previous_kernel = previous_kernel.deepcopy()
                    new_cp: tf.Tensor = \
                        cast(tf.Tensor, tf.constant([(x[0] + x[1]) / 2 for x in xrange],
                                                    shape=[dim, ],
                                                    dtype=global_param.p_dtype))
                    new_kernel_candidates.append(op.ChangePointOperator(
                        dim, [copied_previous_kernel, self.get_base_kernel_by_index(i)], [new_cp]))
            else:
                only_replacements = True

        # If candidate generation by kernel replacement is enabled, extend the current candidate set via that tactic.
        if self.kernel_expression_replacement:
            perm = self.get_candidates_via_base_kernel_replacement(previous_kernel)
            new_kernel_candidates = new_kernel_candidates + perm

        if autogpmi_param.p_optimizable_noise:
            new_kernel_candidates = \
                [op.AdditionOperator(dim, [WhiteNoiseKernel(dim), candidate]) for candidate in new_kernel_candidates]

        return new_kernel_candidates, only_replacements

    def get_candidates_via_base_kernel_replacement(self, kernel: k.Kernel) -> List[k.Kernel]:
        """
        This methods generates new kernel candidates by means of the base kernel replacement tactic (cf. Duvenaud et
        al. (Structure discovery in nonparametric regression through compositional kernel search, in ICML 2013).

        Args:
            kernel: best candidate from the previous iteration of the kernel search

        Returns: set of candidate kernels

        """
        paths = []
        if isinstance(kernel, op.Operator):
            paths = self.get_paths_to_all_base_kernels(kernel, [])

        permutations: List[k.Kernel] = []

        for path in paths:
            for x in range(0, self.get_number_of_base_kernels()):
                perm: k.Kernel = kernel.deepcopy()
                node: k.Kernel = perm

                for r in range(0, len(path) - 1):
                    p = path[r]
                    if isinstance(node, op.Operator):
                        node = node.child_nodes[p]

                assert isinstance(node, op.Operator)

                p = path[len(path) - 1]

                base = self.get_base_kernel_by_index(x)
                if not isinstance(base, type(node.child_nodes[p])):
                    node.replace_child_node(p, base)

                    permutations.append(perm)

        return permutations

    def get_paths_to_all_subexpressions(self, kernel) -> List[List[int]]:
        """
        This is an auxiliary function for generating candidate kernels based on expanding the best candidate from
        the previous iteration of the kernel search by means of concatenating further base kernels. In particular, this
        function supports 'expand_kernel' by delivering all the paths to all feasible subexpressions of the best
        candidate from the previous iteration of the kernel search. A path is an ordered list of indices, which enable
        to traverse a concatenated kernel expression to a certain subexpression.

        Args:
            kernel: best candidate from the previous iteration of the kernel search

        Returns: list of paths, where each path is a list of indices, represented as integers.

        """
        paths: list = []
        if isinstance(kernel, op.Operator):
            for i in range(0, len(kernel.child_nodes)):
                if isinstance(kernel.child_nodes[i], op.Operator):
                    for p in self.get_paths_to_all_subexpressions(kernel.child_nodes[i]):
                        paths = paths + [[i] + p]
                paths = paths + [[i]]

        return paths

    def get_paths_to_all_base_kernels(self, kernel, prev_path: List[int]) -> List[List[int]]:
        """
        This is an auxiliary function for generating candidate kernels based on expanding the best candidate from
        the previous iteration of the kernel search by means of concatenating further base kernels. In particular, this
        function supports 'get_candidates_via_base_kernel_replacement' by delivering all the paths to all constituent
        base kernels of the best candidate from the previous iteration of the kernel search. A path is an ordered
        list of indices, which enable to traverse a concatenated kernel expression to a certain subexpression. Here,
        only those subexpressions are considered, which are base kernels.

        Args:
            kernel: best candidate from the previous iteration of the kernel search

        Returns: list of paths, where each path is a list of indices, represented as integers.

        """
        result_paths: list = []

        for i in range(0, len(kernel.child_nodes)):
            if kernel.child_nodes[i].get_kernel_type() == k.KernelType.OPERATOR:
                sub_result_paths = self.get_paths_to_all_base_kernels(kernel.child_nodes[i], prev_path + [i])

                result_paths = result_paths + sub_result_paths
            else:
                result_paths.append(prev_path + [i])

        return result_paths

    def further_expansion_possible(self, previous_kernel: Kernel) -> bool:
        return self.max_depth > previous_kernel.get_number_base_kernels()


class AbcdStrategy(CksStrategy):
    """
    The ABCD Kernel Expansion Strategy is the one proposed by Lloyd et al. (Automatic construction and natural-language
    description of nonparametric regression models, at AAAI 2014). The basic idea is to generate candidate kernel
    expressions by extending upon the current best performing covariance function / kernel. In doing so, a candidates
    are generated by considering an expansion of every possible subexpressions by means of an operator (Addition,
    Multiplication, and Change Point) and a base kernel.
    """
    def __init__(self, max_depth: int, input_dimensionality: int, kernel_expression_replacement: bool,
                 covariance_function_repository: List[k.Kernel]):
        super(CksStrategy, self).__init__(
            KernelExpansionStrategyType.ABCD, input_dimensionality, kernel_expression_replacement,
            covariance_function_repository)

        self.max_depth: int = max_depth

        # The ABCD Strategy extends upon the CKS kernel expansion strategy by also considering change point operators
        # for concatenating kernels.
        self.operators: \
            List[Union[Type[op.AdditionOperator], Type[op.MultiplicationOperator], Type[op.ChangePointOperator]]] \
            = [op.MultiplicationOperator, op.AdditionOperator, op.ChangePointOperator]


class SkcStrategy(KernelExpansionStrategy):
    """
    The SKC Kernel Expansion Strategy is the one proposed by Kim and Teh (Scaling up the automatic statistician:
    Scalable structure discovery using gaussian processes, at AISTATS 2018). The basic idea is to generate candidate
    kernel expressions by extending upon the current best performing covariance function / kernel. In doing so, a
    candidate is generated by adding or multiplying a further base kernel to the current best performing covariance
    function / kernel.

    The Paper "Scaling up the automatic statistician: Scalable structure discovery using gaussian processes"
    by Kim and Teh (2018) leaves some ambiguity with regards to the kernel expansion mechanisms, since CKS kernel
    expansion is defined there wrong, too. The authors do not explicate a reduction in candidate kernel size in their
    text, but their pseudo code indicates so.
    """

    def __init__(self, max_depth: int, input_dimensionality: int,
                 covariance_function_repository: List[k.Kernel]):
        super(SkcStrategy, self).__init__(
            KernelExpansionStrategyType.CKS, input_dimensionality, False,
            covariance_function_repository)

        self.including_change_point_operator: bool = False
        self.max_depth: int = max_depth

    def get_initial_kernels(self) -> List[k.Kernel]:
        origin_kernels: List[k.Kernel] = []
        for i in range(0, self.get_number_of_base_kernels()):
            origin_kernels.append(self.get_base_kernel_by_index(i))

        dim = self.input_dimensionality
        if autogpmi_param.p_optimizable_noise:
            return [op.AdditionOperator(dim, [WhiteNoiseKernel(dim), origin_kernel])
                    for origin_kernel in origin_kernels]
        else:
            return origin_kernels

    def expand_kernel(
            self, previous_kernel: Kernel, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]]) \
            -> Tuple[List[k.Kernel], bool]:
        if autogpmi_param.p_optimizable_noise:
            previous_kernel = previous_kernel.child_nodes[1]

        expanded_kernels: List[k.Kernel] = []

        if self.max_depth > previous_kernel.get_number_base_kernels():
            for i in range(0, self.get_number_of_base_kernels()):
                copied_pk = previous_kernel.deepcopy()
                expanded_kernels.append(AdditionOperator(
                    input_dimensionality=self.input_dimensionality,
                    child_nodes=[copied_pk, self.get_base_kernel_by_index(i)]))

                copied_pk = previous_kernel.deepcopy()
                expanded_kernels.append(MultiplicationOperator(
                    input_dimensionality=self.input_dimensionality,
                    child_nodes=[copied_pk, self.get_base_kernel_by_index(i)]))

        dim = self.input_dimensionality
        if autogpmi_param.p_optimizable_noise:
            expanded_kernels = \
                [op.AdditionOperator(dim, [WhiteNoiseKernel(dim), expanded_kernel])
                 for expanded_kernel in expanded_kernels]

        return expanded_kernels, len(expanded_kernels) == 0

    def further_expansion_possible(self, previous_kernel: Kernel) -> bool:
        return self.max_depth > previous_kernel.get_number_base_kernels()