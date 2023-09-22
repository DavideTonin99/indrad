import copy

import numpy as np

from ad.AD import AD
from classes.Params import Params
from models.TimeSeries import TimeSeries


class ADQuantile(AD):

    def __init__(self, params: Params):
        """
        Constructor of the class
        @param params: parameters of the execution
        """
        super().__init__(params)

        self.lower_bound = None
        self.upper_bound = None
        self.threshold = None

        # define default parameters for the quantile error threshold
        if self.params.QUANTILE_LOWER_PERCENTAGE:
            self.lower_percentage = self.params.QUANTILE_LOWER_PERCENTAGE
        else:
            self.lower_percentage = 0.05

        if self.params.QUANTILE_UPPER_PERCENTAGE:
            self.upper_percentage = self.params.QUANTILE_UPPER_PERCENTAGE
        else:
            self.upper_percentage = 0.95

        if self.params.QUANTILE_MULTIPLIER:
            self.multiplier = self.params.QUANTILE_MULTIPLIER
        else:
            self.multiplier = 5

    def train(self, ts: np.array) -> None:
        pass

    def calculate_threshold(self, errors: np.array, **kwargs) -> None:
        """
        Calculating the quantile error threshold
        @param errors: errors of the predictions
        @param kwargs: additional arguments
        @return: None
        """
        q1, q2 = np.quantile(errors, [self.lower_percentage, self.upper_percentage], axis=0)
        pc_iqr = q2 - q1
        self.lower_bound = q1 - (self.multiplier * pc_iqr)
        self.upper_bound = q2 + (self.multiplier * pc_iqr)

    def evaluate(self, ts: TimeSeries) -> None:
        pass

    def get_ts_anomalies(self, ts_true: TimeSeries, errors: np.array) -> np.array:
        anomaly_mask = np.logical_or(errors < self.lower_bound, errors > self.upper_bound)

        anomalies = copy.deepcopy(ts_true.data)
        anomalies[np.logical_not(anomaly_mask)] = np.nan

        return anomalies
