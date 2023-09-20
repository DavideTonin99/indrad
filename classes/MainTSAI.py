from classes.Main import Main
from classes.Params import Params
from models.Dataset import Dataset
from models.TimeSeriesUtils import TimeSeriesUtils
from utils.utils import *

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import copy
from tsai.basics import *
import pandas as pd

class MainTSAI(Main):

    DEFAULT_PARAMS = {
        'APPLY_MOVING_AVG': True,
        'MOVING_AVG_STEP': 50,
        'WINDOW_TYPE': 'sliding',
        'WINDOW_SIZE': 2000,
        'WINDOW_STRIDE': 1000,
        'APPLY_PCA': True,
        'PCA_COMPONENTS': 25,
        'NORMALIZER_MODEL': StandardScaler(),
        'THRESHOLD_TYPE': 'quantile',
        'GAUSSIAN_MIXTURE_COMPONENTS': 10,
    }

    def __init__(self, params: dict = {}) -> None:
        params = Params(params)

        for key, value in MainTSAI.DEFAULT_PARAMS.items():
            if not hasattr(params, key):
                setattr(params, key, value)

        self.normalizer_model = params.NORMALIZER_MODEL
        super().__init__(params)

    def train(self, t_list: list = []) -> None:
        """
        Train the model
        :param t_list: list of training time series
        """
        super().pre_train(t_list=t_list)
        if self.params.APPLY_PCA:
            # pca model
            self.pca_model = fit_pca(self.dataset_train_process.ts_data.data, n_components=self.params.PCA_COMPONENTS,
                                    show_plot_variance=False)

        arch_config = dict(
            n_layers=3,  # number of encoder layers
            n_heads=4,  # number of heads
            d_model=16,  # dimension of model
            d_ff=128,  # dimension of fully connected network
            attn_dropout=0.0, # dropout applied to the attention weights
            dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
            patch_len=24,  # length of the patch applied to the time series to create patches
            stride=2,  # stride used when creating patches
            padding_patch=True,  # padding_patch
        )
        df_ts = pd.DataFrame(self.dataset_train_process.ts_data.data)
        X, y = prepare_forecasting_data(df=df_ts, fcst_history=len(self.dataset_train_process.ts_data.data), fcst_horizon=0)
        learn = TSForecaster(X, y, batch_size=16, path="models", pipelines=[], arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=[ShowGraph()])
        print(learn.summary())
        # n_epochs = 100
        # lr_max = 0.0025
        # learn.fit_one_cycle(n_epochs, lr_max=lr_max)

    def evaluate(self, dataset: Dataset, dataset_process: Dataset, mode: str = 'complete', train: bool = False, threshold_params: dict = {}) -> dict:
        """
        Evaluate the model
        :param dataset: dataset to evaluate
        :param mode: mode of the dataset
        :param train: if True, train the model
        :param threshold_params: parameters for the threshold
        """
        if self.params.APPLY_PCA:
            # apply pca on dataset
            dataset_process.pca(model=self.pca_model, mode=mode)

        if train:
            if mode == 'complete':
                # calculate gaussian mixture & mahalanobis distance on train dataset
                if self.params.THRESHOLD_TYPE == 'mahalanobis':
                    self.mahalanobis = {}
                    self.mahalanobis['gm'] = GaussianMixture(
                        n_components=self.params.GAUSSIAN_MIXTURE_COMPONENTS, random_state=0).fit(dataset_process.ts_data.data)
                    self.mahalanobis['covariance_matrix'], self.mahalanobis['inverse_covariance_matrix'] = cov_matrix(
                        dataset_process.ts_data.data)

                    self.mahalanobis_distance_train = []
                    for i in range(self.params.GAUSSIAN_MIXTURE_COMPONENTS):
                        self.mahalanobis_distance_train.append(mahalanobis_dist(
                            self.mahalanobis['inverse_covariance_matrix'], self.mahalanobis['gm'].means_[i], dataset_process.ts_data.data))
                    self.mahalanobis_distance_train = np.median(
                        np.array(self.mahalanobis_distance_train).T)
        else:
            # calculate mahalanobis distance on test dataset
            if self.params.THRESHOLD_TYPE == 'mahalanobis':
                self.mahalanobis_distance_evaluation = {}
                if mode == 'complete':
                    for i in range(self.params.GAUSSIAN_MIXTURE_COMPONENTS):
                        self.mahalanobis_distance_evaluation[dataset_process.name].append(mahalanobis_dist(
                            self.mahalanobis['inverse_covariance_matrix'], self.mahalanobis['gm'].means_[i], dataset_process.ts_data.data))
                    self.mahalanobis_distance_evaluation[dataset_process.name] = np.min(
                        np.array(self.mahalanobis_distance_evaluation[dataset_process.name]).T)
                else:
                    for ts_name in dataset.time_series.keys():
                        self.mahalanobis_distance_evaluation[ts_name] = []
                        for i in range(self.params.GAUSSIAN_MIXTURE_COMPONENTS):
                            self.mahalanobis_distance_evaluation[ts_name].append(mahalanobis_dist(
                                self.mahalanobis['inverse_covariance_matrix'], self.mahalanobis['gm'].means_[i], dataset_process.time_series[ts_name].data))
                        self.mahalanobis_distance_evaluation[ts_name] = np.min(
                            np.array(self.mahalanobis_distance_evaluation[ts_name]).T)

        if self.params.APPLY_PCA:
            dataset_process.pca_inverse(model=self.pca_model, mode=mode)

        # back to original scale
        dataset_process.normalize_inverse(
            normalizer=self.normalizer_model, mode=mode)
        dataset_process.remove_window(step=self.params.WINDOW_STRIDE)

        errors = {}
        if mode == 'complete':
            errors[dataset.name] = compute_errors(
                ts_true=dataset.ts_data, ts_pred=dataset_process.ts_data, abs=False)
        else:
            for ts_name in dataset.time_series.keys():
                errors[ts_name] = compute_errors(
                    ts_true=dataset.time_series[ts_name], ts_pred=dataset_process.time_series[ts_name], abs=False)

        if train:
            if mode == 'complete':
                if self.params.THRESHOLD_TYPE == 'quantile':
                    self.bounds = {'lower': None, 'upper': None}

                    self.bounds = {}
                    self.bounds['lower'], self.bounds['upper'] = compute_quantile_error_threshold(
                        errors=errors[dataset.name], **threshold_params)
                elif self.params.THRESHOLD_TYPE == 'mahalanobis':
                    self.mahalanobis['threshold'] = np.median(
                        np.sum(np.abs(errors[dataset.name]), axis=1), axis=0) * 5   
                elif self.params.THRESHOLD_TYPE == 'score':
                    self.score['threshold'] = np.median(
                        np.sum(np.abs(errors[dataset.name]), axis=1), axis=0) * 5   


    def test(self, t_list: list = [], corruption_params: dict = {}) -> None:
        """
        Test the model
        :param t_list: list of test time series
        """
        super().pre_test(t_list=t_list, corruption_params=corruption_params)

        errors = self.evaluate(
            dataset=self.dataset_test, dataset_process=self.dataset_test_process, mode='single')

        for ts_name in self.dataset_test.time_series.keys():
            self.anomaly(ts_true=self.dataset_test.time_series[ts_name],
                         ts_pred=self.dataset_test_process.time_series[ts_name], errors=errors[ts_name])

        # for ts_name in self.dataset_test.time_series.keys():
        #     plot_ts(f"Figure Test {ts_name}", ts={'start': self.dataset_test.time_series[ts_name].data, 'end': self.dataset_test_process.time_series[ts_name].data}, features=TimeSeriesUtils.FEATURES,
        #             n_rows=TimeSeriesUtils.N_JOINTS, n_cols=len(TimeSeriesUtils.FEATURES), figsize=(15, 5), colors={'start': 'black', 'end': 'orange'})
        #     plt.show()

    def anomaly(self, ts_true: TimeSeries, ts_pred: TimeSeries, errors: np.array, show_plot: bool = True) -> None:
        if self.params.THRESHOLD_TYPE == 'quantile':
            anomalies_mask = ((errors > self.bounds['upper']) | (
                errors < self.bounds['lower'])) == True
            
            anomalies = copy.deepcopy(ts_true.data)
            anomalies[np.logical_not(anomalies_mask)] = np.nan

            plot_ts(f"Figure Test {ts_true.name} with anomalies", ts={'ts': ts_true.data, 'anomaly': anomalies}, features=TimeSeriesUtils.FEATURES,
                    n_rows=TimeSeriesUtils.N_JOINTS, n_cols=len(TimeSeriesUtils.FEATURES), figsize=(15, 5), colors={'ts': 'black', 'anomaly': 'red'})
            plt.show()

        elif self.params.THRESHOLD_TYPE == 'mahalanobis':
            anom_index = np.where(
                (self.mahalanobis_distance_evaluation > self.mahalanobis_distance_train) == True)

            # plot_anomaly_score(ts_true.name, self.mahalanobis_distance_evaluation, None, mahalanobis_distance_train)

            anomaly_score = np.sum(np.abs(errors), axis=1)

            anomalies = (anomaly_score > self.mahalanobis['threshold']) == True
            non_anomalies = (
                anomaly_score <= self.mahalanobis['threshold']) != True
            # plot_anomaly_score(ts_true.name, anomaly_score, anomalies, self.mahalanobis['threshold'])

            if len(anom_index[0]) > 0:
                for index in range(len(self.mahalanobis_distance_evaluation)):
                    if index not in anom_index[0]:
                        start = index * self.params.WINDOW_STRIDE
                        end = index * self.params.WINDOW_STRIDE + self.params.WINDOW_STRIDE
                        anomalies[start:end, :] = False
                        non_anomalies[start:end, :] = True
                ts_anomalies = ts_true[anomalies.to_numpy()]
            else:
                for index in range(len(self.mahalanobis_distance_evaluation)):
                    if index not in anom_index[0]:
                        anomalies.iloc[index*self.params.WINDOW_STRIDE:index *
                                       self.params.WINDOW_STRIDE+self.params.WINDOW_STRIDE, :] = False
                        non_anomalies.iloc[index*self.params.WINDOW_STRIDE:index *
                                           self.params.WINDOW_STRIDE+self.params.WINDOW_STRIDE, :] = True
                ts_anomalies = np.array([])

        elif self.params.THRESHOLD_TYPE == 'score':
            anomaly_score = np.sum(np.abs(errors), axis=1)
            anomalies = (anomaly_score > self.score['threshold']) == True

            trajectory_anomalies = ts_true.data[anomalies]

            # plot_anomaly_score(ts_true.name, anomaly_score, anomalies, self.score['threshold'])


    def run(self, train_list: list = None, test_list: list = None, corruption_params: dict = {}) -> None:
        """
        Run the model
        :param training_list: list of training time series
        :param test_list: list of test time series
        :param train: if True, train the model
        :param test: if True, test the model
        """
        if train_list is not None and len(train_list) > 0:
            self.train(t_list=train_list)
            self.evaluate(dataset=self.dataset_train, dataset_process=self.dataset_train_process,
                          mode='complete', train=True, threshold_params={'lower_perc': 0.05, 'upper_perc': 0.95})
        if test_list is not None and len(test_list) > 0:
            self.test(t_list=test_list, corruption_params=corruption_params)
