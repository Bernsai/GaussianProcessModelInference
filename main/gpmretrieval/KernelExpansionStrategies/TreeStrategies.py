import gpbasics.global_parameters as global_param

global_param.ensure_init()

from typing import List, Tuple, cast, Union
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.BaseKernels as bk
import gpbasics.KernelBasics.Kernel as k
import gpbasics.KernelBasics.Operators as op
from gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy import \
    KernelExpansionStrategyType, KernelExpansionStrategy
import gpmretrieval.autogpmr_parameters as autogpmi_param


class SumOfProductsExpansionStrategy(KernelExpansionStrategy):
    """
    The Sum-of-Products Kernel Expansion Strategy outlined by Berns and Beecks (Complexity-adaptive gaussian process
    model inference for large-scale data, at SDM 2021). The basic idea is to generate candidate kernel
    expressions by extending upon the current best performing covariance function / kernel. Here, every candidate kernel
    adheres to the sum-of-products form and extension of kernels are inherently implemented so, too.
    """
    def __init__(self, max_npo: int, max_depth: int, input_dimensionality: int, kernel_expression_replacement: bool,
                 covariance_function_repository: List[k.Kernel]):
        super(SumOfProductsExpansionStrategy, self).__init__(
            KernelExpansionStrategyType.SumOfProducts, input_dimensionality, kernel_expression_replacement,
            covariance_function_repository)

        # max npo = Maximum amount of child Nodes Per Operator
        self.max_npo: int = max_npo

        # maximum amount of constituent base kernels in a kernel
        self.max_depth: int = max_depth

    def get_initial_kernels(self) -> List[op.AdditionOperator]:
        origin_kernels: List[op.AdditionOperator] = []
        for i in range(self.get_number_of_base_kernels()):
            base: k.Kernel = self.get_base_kernel_by_index(i)
            encapsulated: op.AdditionOperator = \
                op.AdditionOperator(self.input_dimensionality,
                                    [op.MultiplicationOperator(self.input_dimensionality, [base])])

            origin_kernels.append(encapsulated)

        dim = self.input_dimensionality
        if autogpmi_param.p_optimizable_noise:
            return [op.AdditionOperator(dim, [bk.WhiteNoiseKernel(dim), origin_kernel])
                    for origin_kernel in origin_kernels]
        else:
            return origin_kernels

    def get_permutations_for_additive_level(self, kernel_tree: op.AdditionOperator) -> List[op.AdditionOperator]:
        """
        This method produces candidate kernel expressions by concatenating further base kernels on the additive level
        of the sum-of-product kernel expressions, which are solely considered by the basic hierarchical kernel expansion
        strategy.

        Args:
            kernel_tree: kernel expression to extend upon

        Returns: list of candidate kernels in sum-of-products form

        """
        permutations: List[op.AdditionOperator] = []

        if (self.max_npo is not None and kernel_tree.get_number_of_child_nodes() >= self.max_npo) or \
                (self.max_depth is not None and kernel_tree.get_number_base_kernels() >= self.max_depth):
            return permutations

        for i in range(self.get_number_of_base_kernels()):
            copied_kernel_tree: op.AdditionOperator = kernel_tree.deepcopy()

            base: k.Kernel = self.get_base_kernel_by_index(i)

            encapsulated = op.MultiplicationOperator(self.input_dimensionality, [base])

            copied_kernel_tree.add_kernel(encapsulated)

            permutations.append(copied_kernel_tree)

        return permutations

    def get_permutations_for_multiplicative_level(self, kernel_tree: op.AdditionOperator, path: int) \
            -> List[op.AdditionOperator]:
        """
        This method produces candidate kernel expressions by concatenating further base kernels on the multiplicative
        level of the sum-of-product kernel expressions, which are solely considered by the basic hierarchical kernel
        expansion strategy.

        Args:
            kernel_tree: kernel expression to extend upon

        Returns: list of candidate kernels in sum-of-products form

        """
        permutations: List[op.AdditionOperator] = []

        if (self.max_npo is not None and kernel_tree.child_nodes[path].get_number_of_child_nodes() >= self.max_npo) or \
                (self.max_depth is not None and kernel_tree.get_number_base_kernels() >= self.max_depth):
            return permutations

        for i in range(self.get_number_of_base_kernels()):
            copied_kernel_tree: op.AdditionOperator = kernel_tree.deepcopy()

            mul_op: op.MultiplicationOperator = cast(op.MultiplicationOperator, copied_kernel_tree.child_nodes[path])

            base: k.Kernel = self.get_base_kernel_by_index(i)

            mul_op.add_kernel(base)

            permutations.append(copied_kernel_tree)

        return permutations

    def get_candidates_via_base_kernel_replacement(self, kernel_tree: op.AdditionOperator, path: int) \
            -> List[op.AdditionOperator]:
        """
        This method generates new kernel candidates by means of the base kernel replacement tactic (cf. Duvenaud et
        al. (Structure discovery in nonparametric regression through compositional kernel search, in ICML 2013). This
        tactic is actually not used for algorithms using the Sum-of-Products kernel expansion strategies. Still, it is
        implemented for testing purposes.

        Args:
            kernel_tree: best candidate from the previous iteration of the kernel search
            path: at which part to extend via base kernel replacement

        Returns: set of candidate kernels

        """
        permutations = []
        for x in range(self.get_number_of_base_kernels()):
            assert isinstance(kernel_tree.child_nodes[path], op.Operator)
            for y in range(len(kernel_tree.child_nodes[path].child_nodes)):
                copied_kernel_tree: op.AdditionOperator = kernel_tree.deepcopy()
                mul_op: op.MultiplicationOperator = \
                    cast(op.MultiplicationOperator, copied_kernel_tree.child_nodes[path])

                base: bk.BaseKernel = self.get_base_kernel_by_index(x)
                if not isinstance(base, type(mul_op.child_nodes[y])):
                    mul_op.replace_child_node(y, base)

                    permutations.append(copied_kernel_tree)

        return permutations

    def further_expansion_possible(self, previous_kernel: op.AdditionOperator) -> bool:
        if autogpmi_param.p_optimizable_noise:
            previous_kernel = previous_kernel.child_nodes[1]

        if self.max_depth is not None and previous_kernel.get_number_base_kernels() >= self.max_depth:
            return False

        if self.max_npo is None or len(previous_kernel.child_nodes) < self.max_npo:
            return True

        for i in range(len(previous_kernel.child_nodes)):
            if self.max_npo is None or len(previous_kernel.child_nodes[i].child_nodes) < self.max_npo:
                return True

        return False

    def expand_kernel(self, previous_kernel: k.Kernel,
                      data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]]) \
            -> Tuple[List[op.AdditionOperator], bool]:
        if autogpmi_param.p_optimizable_noise:
            previous_kernel = previous_kernel.child_nodes[1]

        assert isinstance(previous_kernel, op.AdditionOperator)

        new_kernels: List[op.Operator] = []
        only_replacements: bool = True
        if len(previous_kernel.child_nodes) < self.max_npo:
            only_replacements = False
            perms: List[op.Operator] = self.get_permutations_for_additive_level(previous_kernel)
            new_kernels = new_kernels + perms

        for mul_i in range(len(previous_kernel.child_nodes)):
            if len(previous_kernel.child_nodes[mul_i].child_nodes) < self.max_npo:
                only_replacements = False
                perms = self.get_permutations_for_multiplicative_level(previous_kernel, mul_i)
                new_kernels = new_kernels + perms

            if self.kernel_expression_replacement is not None \
                    and self.kernel_expression_replacement:
                replace = self.get_candidates_via_base_kernel_replacement(previous_kernel, mul_i)
                new_kernels = new_kernels + replace

        dim = self.input_dimensionality
        if autogpmi_param.p_optimizable_noise:
            new_kernels = \
                [op.AdditionOperator(dim, [bk.WhiteNoiseKernel(dim), candidate]) for candidate in new_kernels]

        return new_kernels, only_replacements
