import numpy as np

from ad.AD import AD
from classes.Params import Params


class ADScore(AD):

    def __init__(self, params: Params):
        """
        Constructor of the class
        @param params: parameters of the execution
        """
        super().__init__(params)

        self.threshold = None
        self.anomaly_score = None
        self.anomaly_mask = None
        self.anomalies = None

        # define default parameters for the score error threshold
        if self.params.QUANTILE_MULTIPLIER:
            self.multiplier = self.params.QUANTILE_MULTIPLIER
        else:
            self.multiplier = 5

    def train(self, ts_data: np.array) -> None:
        pass

    def calculate_threshold(self, errors: np.array, **kwargs) -> None:
        """
        Calculating the quantile error threshold
        @param errors: errors of the predictions
        @param kwargs: additional arguments
        @return: None
        """
        self.threshold = np.median(np.sum(np.abs(errors), axis=1), axis=0) * self.multiplier

    def evaluate(self, ts_data: np.array) -> None:
        pass

    def get_ts_anomalies(self, ts_data: np.array, errors: np.array) -> np.array:
        self.anomaly_score = np.sum(np.abs(errors), axis=1)
        self.anomaly_mask = True == (self.anomaly_score > self.threshold)

        self.anomalies = ts_data[self.anomaly_mask]

        return self.anomalies
