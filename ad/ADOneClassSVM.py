import pickle

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

from ad.AD import AD
from classes.Params import Params
from utils.utils import cov_matrix, mahalanobis_dist


class ADOneClassSVM(AD):

    def __init__(self, params: Params, **kwargs):
        super().__init__(params)

        if 'load_filename' in kwargs and kwargs['load_filename']:
            with open(kwargs['load_filename'], 'rb') as file:
                self.model = pickle.load(file)
        else:
            self.model = OneClassSVM(kernel=self.params.AD_OCSVM_KERNEL, nu=self.params.AD_OCSVM_NU,
                                     gamma=self.params.AD_OCSVM_GAMMA)
        self.threshold = None
        self.anomaly_mask = None
        self.anomaly_score = None
        self.anomalies = None

    def train(self, ts_data: np.array) -> None:
        """
        Training the model
        @param ts_data: errors of pca reconstruction
        @return:
        """
        self.model = self.model.fit(ts_data)

    def calculate_threshold(self, errors: np.array, **kwargs) -> None:
        self.threshold = np.median(np.abs(errors), axis=0)

    def evaluate(self, ts_data: np.array) -> None:
        pass

    def get_ts_anomalies(self, ts_data: np.array, errors: np.array) -> np.array:
        one_class_predict = self.model.predict(errors)
        anom_index = np.where(one_class_predict == -1)[0]
        self.anomalies = ts_data[anom_index]

        return self.anomalies
