import gpbasics.global_parameters as global_param

global_param.ensure_init()

import logging
from enum import Enum
from typing import List
import ruptures as rpt
import gpbasics.DataHandling.DataInput as di
import tensorflow as tf


# the change point detection mechanisms provided via the ruptures library
# (cf. https://centre-borelli.github.io/ruptures-docs/) are provided here.
class BatchChangePointDetectionType(Enum):
    PELT = 101
    Binary = 102
    BottomUp = 103
    WindowBased = 104


class BatchChangePointDetection:
    """
    This is a general class defining the structure for those implementations, which return all the change points of a
    dataset at once. This notion of _Batch Change Point Detection_ has a certain overlap with the concept of _Offline
    Change Point Detection_, but we do not regard these two concepts as interchangeable.
    """

    def get_change_points(self, data_input: di.DataInput, number_of_changepoints: int = None) -> List[int]:
        """
        This method performs change point detection on the provided data and returns a list of indices, which identify
        those position in the training data of the data input, where change points were detected.
        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points
            number_of_changepoints: if required, the amount of change points one wants to recover can be specified.

        Returns: List of indices identifying the respective change points in the training data

        """
        pass


class BatchWrapperRuptures(BatchChangePointDetection):
    def __init__(self, detection_type: BatchChangePointDetectionType = BatchChangePointDetectionType.Binary):
        super(BatchWrapperRuptures, self).__init__()
        self.type: BatchChangePointDetectionType = detection_type
        self.absolute_minimum_segment_size = 120

        # The model parameter in ruptures defines the used cost function
        # options: "l1", "l2", "rbf", "linear", "normal", "ar"
        # cf. https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costl2/
        self.model = "l1"
        logging.info(f"Used model for change point detection: '{self.model}'.")

    def get_change_points(self, data_input: di.DataInput, number_of_changepoints: int = None) -> List[int]:
        if self.type is BatchChangePointDetectionType.PELT:
            if number_of_changepoints is not None:
                logging.warning("PELT does not support to give the number of changepoints in advance.")
            return self.get_change_points_pelt(data_input)
        elif self.type is BatchChangePointDetectionType.Binary:
            return self.get_change_points_binary(data_input, number_of_changepoints)
        elif self.type is BatchChangePointDetectionType.BottomUp:
            return self.get_change_points_bottom_up(data_input, number_of_changepoints)
        elif self.type is BatchChangePointDetectionType.WindowBased:
            return self.get_change_points_window_based(data_input, number_of_changepoints)
        else:
            logging.warning(f"Invalid change point detection type: '{self.type}'.")
            return None

    @staticmethod
    def get_penalty(data_input):
        """
        This method calculates a custom penalty inspired by the one employed in
        https://centre-borelli.github.io/ruptures-docs/user-guide/detection/binseg/. This penalty is used, when the
        number of change points one searches for is not provided.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points

        Returns: a scalar, float indicating the respective penalty

        """
        sigma = tf.math.reduce_std(data_input.data_x_train)
        penalty = tf.math.log(tf.cast(data_input.n_train, dtype=global_param.p_dtype)) * \
                  data_input.get_input_dimensionality() * tf.square(sigma)
        if data_input.n_train < 100000:
            return float(penalty / 3)
        else:
            return float(penalty / 5)

    @staticmethod
    def get_signal(data_input) -> List[float]:
        """
        This is an auxiliary function used to retrieve the training target data (data_y_train), which is already
        normalized by subtracting the mean function.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points

        Returns: target training data - mean function evaluated at input training data

        """
        assert data_input.get_input_dimensionality() == 1, \
            "Changepoint Detection is only viable for data with one-dimensional input data."

        return tf.reshape(
            data_input.get_detrended_y_train(), shape=[data_input.data_y_train.shape[0], ]).numpy()

    def get_change_points_pelt(self, data_input: di.DataInput) -> List[int]:
        """
        Change Point Detection Mechanism: PELT:
        Killick, R., Fearnhead, P., & Eckley, I. (2012). Optimal detection of changepoints with a linear computational
        cost. Journal of the American Statistical Association, 107(500), 1590–1598.

        This method performs change point detection on the provided data and returns a list of indices, which identify
        those position in the training data of the data input, where change points were detected.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points

        Returns: List of indices identifying the respective change points in the training data

        """
        penalty = self.get_penalty(data_input)
        minimum_segment_size = \
            min(self.absolute_minimum_segment_size, 500)
        cps = rpt.Pelt(model=self.model, min_size=minimum_segment_size).fit_predict(
            pen=penalty, signal=self.get_signal(data_input))

        return cps[0:len(cps) - 1]

    def get_change_points_binary(self, data_input: di.DataInput, number_of_changepoints: int = None) -> List[int]:
        """
        Change Point Detection Mechanism: Binary Segmentation:
        Bai, J. (1997). Estimating multiple breaks one at a time. Econometric Theory, 13(3), 315–352.

        Fryzlewicz, P. (2014). Wild binary segmentation for multiple change-point detection.
        The Annals of Statistics, 42(6), 2243–2281.

        This method performs change point detection on the provided data and returns a list of indices, which identify
        those position in the training data of the data input, where change points were detected.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points
            number_of_changepoints: if required, the amount of change points one wants to recover can be specified.

        Returns: List of indices identifying the respective change points in the training data

        """
        if number_of_changepoints is None:
            penalty = self.get_penalty(data_input)
        else:
            penalty = None

        minimum_segment_size = \
            min(self.absolute_minimum_segment_size, 500)

        cps = rpt.Binseg(model=self.model, min_size=minimum_segment_size).fit_predict(
            pen=penalty, n_bkps=number_of_changepoints, signal=self.get_signal(data_input))

        return cps[0:len(cps) - 1]

    def get_change_points_bottom_up(self, data_input: di.DataInput, number_of_changepoints: int = None) -> List[int]:
        """
        Change Point Detection Mechanism: Bottom-Up Segmentation:
        Keogh, E., Chu, S., Hart, D., & Pazzani, M. (2001). An online algorithm for segmenting time series.
        Proceedings of the IEEE International Conference on Data Mining (ICDM), 289–296.

        Fryzlewicz, P. (2007). Unbalanced Haar technique for nonparametric function estimation.
        Journal of the American Statistical Association, 102(480), 1318–1327.

        This method performs change point detection on the provided data and returns a list of indices, which identify
        those position in the training data of the data input, where change points were detected.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points
            number_of_changepoints: if required, the amount of change points one wants to recover can be specified.

        Returns: List of indices identifying the respective change points in the training data

        """

        if number_of_changepoints is None:
            penalty = self.get_penalty(data_input)
        else:
            penalty = None

        minimum_segment_size = \
            min(self.absolute_minimum_segment_size, 500)

        cps = rpt.BottomUp(model=self.model, min_size=minimum_segment_size).fit_predict(
            pen=penalty, n_bkps=number_of_changepoints, signal=self.get_signal(data_input))

        return cps[0:len(cps) - 1]

    def get_change_points_window_based(self, data_input: di.DataInput, number_of_changepoints: int = None) -> List[int]:
        """
        Change Point Detection Mechanism: Window-based Change Point Detection:
        C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods.
        Signal Processing, 167:107299, 2020.

        This method performs change point detection on the provided data and returns a list of indices, which identify
        those position in the training data of the data input, where change points were detected.

        Args:
            data_input: DataInput, whose target training data (data_y_train) is searched for change points
            number_of_changepoints: if required, the amount of change points one wants to recover can be specified.

        Returns: List of indices identifying the respective change points in the training data

        """

        if number_of_changepoints is None:
            penalty = self.get_penalty(data_input)
        else:
            penalty = None

        minimum_segment_size = \
            min(self.absolute_minimum_segment_size, 500)

        cps = rpt.Window(model=self.model, min_size=minimum_segment_size).fit_predict(
            pen=penalty, n_bkps=number_of_changepoints, signal=self.get_signal(data_input))

        return cps[0:len(cps) - 1]
