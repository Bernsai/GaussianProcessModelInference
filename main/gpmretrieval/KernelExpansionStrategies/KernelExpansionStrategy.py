import gpbasics.global_parameters as global_param

global_param.ensure_init()

from enum import Enum
from typing import List, Union, Tuple
from gpbasics.KernelBasics.Kernel import Kernel
from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput


class KernelExpansionStrategyType(Enum):
    CKS = 1
    ABCD = 2
    SKC = 3
    SumOfProducts = 4


class KernelExpansionStrategy:
    """
    A Kernel Expansion Strategy is the set of rules, which are applied to generate candidate kernels to be employed as a
    covariance function for a Gaussian process. This candidate generation is a key part of the local search procedure of
    Gaussian process model inference.
    """

    def __init__(self, strategy_type: KernelExpansionStrategyType, input_dimensionality: int,
                 kernel_expression_replacement: bool, covariance_function_repository: List[Kernel]):
        self.strategy_type: KernelExpansionStrategyType = strategy_type
        self.input_dimensionality = input_dimensionality

        # One tactic of kernel expansion strategies is generating candidate kernels by replacing constituent base
        # kernels of the current best candidate by any other considered base kernel. This tactic hugely expands on the
        # considered search space, which (i) might increase expressiveness of the resulting kernel, but also (ii)
        # increases runtime by a substantial portion. Therefore, this tactic can be enabled separately.
        self.kernel_expression_replacement: bool = kernel_expression_replacement

        # If provided, candidate kernels can be constructed from predefined kernels coming in a 'covariance
        # function repository'. In this case, kernels in that repository represent particular behaviors of interest,
        # with predefined hyperparameters. Therefore, these would not be optimized, but only constructed and
        # subsequently evaluated.
        self.covariance_function_repository: List[Kernel] = covariance_function_repository
        self.use_covariance_function_repository: bool = len(self.covariance_function_repository) > 0

    def get_number_of_base_kernels(self) -> int:
        """
        This functions returns the amount of considered base kernels. It provides access to that information agnostic 
        with regards to the potential usage of a covariance function repository.
        Returns: 

        """
        if self.use_covariance_function_repository:
            return len(self.covariance_function_repository)
        else:
            return len(global_param.p_used_base_kernel)

    def get_base_kernel_by_index(self, idx: int) -> Kernel:
        """
        This method returns a basic kernel, from which a concatenated kernel may be constructed. Such a base kernel is
        either a base kernel or an element of the covariance function repository (if provided). It provides access to
        that information agnostic with regards to the potential usage of a covariance function repository.
        Args:
            idx: index of the kernel, to be returned

        Returns: requested kernel

        """
        if self.use_covariance_function_repository:
            return self.covariance_function_repository[idx]
        else:
            return global_param.p_used_base_kernel[idx](self.input_dimensionality)

    def get_initial_kernels(self) -> List[Kernel]:
        """
        This method returns the initial candidate set for the respective kernel search, i.e. when no previous
        candidate is known.

        Returns: Set of initial kernels of type Kernel

        """
        pass

    def expand_kernel(self, previous_kernel: Kernel, data_input: Union[AbstractDataInput, List[AbstractDataInput]]) \
            -> Tuple[List[Kernel], bool]:
        """
        This method returns a set of candidate kernels, which have been generated based on the best candidate from the
        previous iteration of the kernel search.

        Args:
            previous_kernel: best candidate from the previous iteration of the kernel search
            data_input: considered data input for the kernel search

        Returns: a tuple. First element of that tuple contains the list of new candidate kernels, while the second
        element is a boolean indicating, whether this candidate set has only been generated via the tactic of kernel
        replacement (True).

        """
        pass

    def further_expansion_possible(self, previous_kernel: Kernel) -> bool:
        """
        Assesses, whether under the given constraints for the kernel search a further iteration of the kernel search and
        thus a further expansion of the current best kernel is possible.

        Args:
            previous_kernel: best candidate from the previous iteration of the kernel search

        Returns: True: further expansion possible, False: no further expansion possible except for base kernel
        replacement.

        """
        return True
