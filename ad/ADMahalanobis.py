import numpy as np
from sklearn.mixture import GaussianMixture

from ad.AD import AD
from classes.Params import Params
from utils.utils import cov_matrix, mahalanobis_dist


class ADMahalanobis(AD):

    def __init__(self, params: Params):
        super().__init__(params)

        self.gm = None
        self.covariance_matrix = None
        self.inverse_covariance_matrix = None
        self.mahalanobis_distance_train = None
        self.mahalanobis_distance = None
        self.threshold = None

        # define default parameters for the mahalanobis error threshold
        if self.params.GAUSSIAN_MIXTURE_COMPONENTS:
            self.gaussian_mixture_components = self.params.GAUSSIAN_MIXTURE_COMPONENTS
        else:
            self.gaussian_mixture_components = 10

        if self.params.MAHALANOBIS_MULTIPLIER:
            self.multiplier = self.params.MAHALANOBIS_MULTIPLIER
        else:
            self.multiplier = self.params.MAHALANOBIS_MULTIPLIER if self.params.MAHALANOBIS_MULTIPLIER else 5

    def train(self, ts_data: np.array) -> None:
        self.gm = GaussianMixture(n_components=self.gaussian_mixture_components, random_state=0).fit(
            ts_data)
        self.covariance_matrix, self.inverse_covariance_matrix = cov_matrix(ts_data)

        self.mahalanobis_distance_train = []
        for i in range(self.gaussian_mixture_components):
            self.mahalanobis_distance_train.append(
                mahalanobis_dist(self.inverse_covariance_matrix, self.gm.means_[i], ts_data))
        self.mahalanobis_distance_train = np.median(np.array(self.mahalanobis_distance_train).T)

    def calculate_threshold(self, errors: np.array, **kwargs) -> None:
        self.threshold = np.median(np.sum(np.abs(errors), axis=1), axis=0) * self.multiplier

    def evaluate(self, ts_data: np.array) -> None:
        self.mahalanobis_distance = []
        for i in range(self.params.GAUSSIAN_MIXTURE_COMPONENTS):
            self.mahalanobis_distance.append(
                mahalanobis_dist(self.inverse_covariance_matrix, self.gm.means_[i], ts_data))
        self.mahalanobis_distance = np.min(np.array(self.mahalanobis_distance).T, axis=1)

    def get_ts_anomalies(self, ts_data: np.array, errors: np.array) -> np.array:
        anom_index = np.where(self.mahalanobis_distance > self.mahalanobis_distance_train)[0]
        anomaly_score = np.sum(np.abs(errors), axis=1)

        anomaly_mask = True == (anomaly_score > self.threshold)

        if len(anom_index) > 0:
            for index in range(len(self.mahalanobis_distance)):
                if index in anom_index:
                    start = index * self.params.WINDOW_STRIDE
                    end = index * self.params.WINDOW_STRIDE + self.params.WINDOW_STRIDE
                    anomaly_mask[start:end] = False
            anomalies = ts_data[anomaly_mask]
        else:
            for index in range(len(self.mahalanobis_distance)):
                if index not in anom_index:
                    start = index * self.params.WINDOW_STRIDE
                    end = index * self.params.WINDOW_STRIDE + self.params.WINDOW_STRIDE
                    anomaly_mask[start:end] = False
            anomalies = np.array([])

        return anomalies
