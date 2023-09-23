from abc import abstractmethod

import numpy as np

from classes.Params import Params


class AD:

    def __init__(self, params: Params):
        self.params = params

    def get(self, attr: str):
        result = None
        if hasattr(self, attr):
            result = getattr(self, attr)
        return result

    @abstractmethod
    def train(self, ts_data: np.array) -> None:
        raise NotImplementedError("train method not implemented")

    @abstractmethod
    def calculate_threshold(self, errors: np.array, **kwargs) -> None:
        raise NotImplementedError("calculate_threshold method not implemented")

    @abstractmethod
    def evaluate(self, ts_data: np.array) -> None:
        """
        Evaluate the model with the data of the time series
        @param ts_data: time series data
        """
        pass

    def get_ts_anomalies(self, ts_data: np.array, errors: np.array) -> np.array:
        """
        Get the anomalies of the time series
        @param ts_data: time series with the true values
        @param errors: errors of the predictions
        @return:
        """
        pass
