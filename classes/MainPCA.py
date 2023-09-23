from ad.ADMahalanobis import ADMahalanobis
from ad.ADOneClassSVM import ADOneClassSVM
from ad.ADQuantile import ADQuantile
from ad.ADScore import ADScore
from classes.Main import Main
from classes.Params import Params
from models.TimeSeriesUtils import TimeSeriesUtils
from utils.Plot import Plot
from utils.utils import *

from sklearn.preprocessing import StandardScaler


class MainPCA(Main):
    """
    Main class implementation for PCA
    """

    DEFAULT_PARAMS = {
        'APPLY_MOVING_AVG': True,
        'MOVING_AVG_STEP': 50,
        'WINDOW_TYPE': 'sliding',
        'WINDOW_SIZE': 2000,
        'WINDOW_STRIDE': 1000,
        'PCA_COMPONENTS': 25,
        'NORMALIZER_MODEL': StandardScaler(),
        'THRESHOLD_TYPE': 'quantile',
        'GAUSSIAN_MIXTURE_COMPONENTS': 10,
        'OCSVM_KERNEL': 'rbf',
        'OCSVM_GAMMA': 0.001,
        'OCSVM_NU': 0.03,
        'QUANTILE_LOWER_PERCENTAGE': 0.01,
        'QUANTILE_UPPER_PERCENTAGE': 0.99,
        'QUANTILE_MULTIPLIER': 5,
        'MAHALANOBIS_MULTIPLIER': 5
    }

    def __init__(self, params: Params = None) -> None:
        """
        Parameters
        ----------
        @param params: Dictionary of parameters to be set as attributes
        """
        if params is None:
            params = Params({})
        super().__init__(params)

        for key, value in MainPCA.DEFAULT_PARAMS.items():
            if not hasattr(self.params, key):
                setattr(self.params, key, value)

        self.pca_model = None

        # define anomaly detection model
        self.ad_model = None
        if self.params.THRESHOLD_TYPE == 'quantile':
            self.ad_model = ADQuantile(params=params)
        elif self.params.THRESHOLD_TYPE == 'mahalanobis':
            self.ad_model = ADMahalanobis(params=params)
        elif self.params.THRESHOLD_TYPE == 'score':
            self.ad_model = ADScore(params=params)
        elif self.params.THRESHOLD_TYPE == 'ocsvm':
            self.ad_model = ADOneClassSVM(params=params)

    def train(self, t_list: list = None, threshold_params: dict = None, plot_stats: bool = False) -> None:
        if t_list is None:
            t_list = []
        if threshold_params is None:
            threshold_params = {}

        super().pre_train(t_list=t_list)

        # pca model
        self.pca_model = fit_pca(self.dataset_train_process.ts_data.data, n_components=self.params.PCA_COMPONENTS,
                                 show_plot_variance=False)
        self.dataset_train_process.pca(model=self.pca_model)

        # calculate gaussian mixture & mahalanobis distance on train time series
        if self.params.THRESHOLD_TYPE == 'mahalanobis':
            self.ad_model.train(ts_data=self.dataset_train_process.ts_data.data)

        self.dataset_train_process.pca_inverse(model=self.pca_model)

        # back to original scale
        self.dataset_train_process.normalize_inverse(normalizer=self.normalizer_model)
        self.dataset_train_process.remove_window(step=self.params.WINDOW_STRIDE)

        errors = compute_errors(ts_true=self.dataset_train.ts_data,
                                ts_pred=self.dataset_train_process.ts_data, abs=False)

        if plot_stats:
            plot = Plot()
            for plot_type in ['distribution', 'quantile']:
                plot.stats(title=f"Figure Train {plot_type} errors", ts_data=errors, features=TimeSeriesUtils.FEATURES,
                           n_rows=TimeSeriesUtils.N_JOINTS, n_cols=len(TimeSeriesUtils.FEATURES), plot_type=plot_type,
                           show=True)

        if self.params.THRESHOLD_TYPE == 'ocsvm':
            self.ad_model.train(ts_data=errors)

        threshold_params['errors'] = errors
        self.ad_model.calculate_threshold(**threshold_params)

    def test(self, t_list: list = None, corruption_params: dict = None, show_plot: bool = True) -> None:
        super().pre_test(t_list=t_list, corruption_params=corruption_params, ad_model=self.ad_model)

        self.dataset_test_process.pca(model=self.pca_model)

        if self.params.THRESHOLD_TYPE == 'mahalanobis':
            for ts_name in self.dataset_test_process.time_series.keys():
                self.dataset_test_process.time_series[ts_name].evaluate_ad_model()

        self.dataset_test_process.pca_inverse(model=self.pca_model)

        self.dataset_test_process.normalize_inverse(normalizer=self.normalizer_model)
        self.dataset_test_process.remove_window(step=self.params.WINDOW_STRIDE)

        plot = Plot()
        for ts_name in self.dataset_test.time_series.keys():
            plot.ts(f"Figure Test {ts_name}", time_series={'ts_true': self.dataset_test.time_series[ts_name].data,
                                                           'ts_pred': self.dataset_test_process.time_series[
                                                               ts_name].data},
                    features=TimeSeriesUtils.FEATURES,
                    n_rows=TimeSeriesUtils.N_JOINTS, n_cols=len(TimeSeriesUtils.FEATURES), show=show_plot)

            self.dataset_test.time_series[ts_name].set_ad_model(self.dataset_test_process.time_series[ts_name].ad_model)

            self.errors[ts_name] = compute_errors(
                ts_true=self.dataset_test.time_series[ts_name], ts_pred=self.dataset_test_process.time_series[ts_name],
                abs=False)
            self.anomaly(ts_true=self.dataset_test.time_series[ts_name], errors=self.errors[ts_name],
                         show_plot=show_plot)

    def predict(self, ts: TimeSeries, ts_process: TimeSeries, threshold_params: dict = None) -> None:
        pass

    def anomaly(self, ts_true: TimeSeries, errors: np.array, show_plot: bool = True) -> None:
        anomalies = ts_true.ad_model.get_ts_anomalies(ts_data=ts_true.data, errors=errors)
        anomaly_mask = ts_true.ad_model.get('anomaly_mask')
        anomaly_score = ts_true.ad_model.get('anomaly_score')
        threshold = ts_true.ad_model.get('threshold')

        plot = Plot()
        if self.params.THRESHOLD_TYPE in ['mahalanobis', 'score']:
            plot.anomaly_score(ts_name=ts_true.name, anomaly_score=anomaly_score, anomaly_mask=anomaly_mask,
                               threshold=threshold, show=show_plot)

        plot.ts(title=f"Figure Test {ts_true.name} with anomalies",
                time_series={'ts_ok': ts_true.data, 'anomaly': anomalies},
                features=TimeSeriesUtils.FEATURES,
                n_rows=TimeSeriesUtils.N_JOINTS, n_cols=len(TimeSeriesUtils.FEATURES), show=show_plot)

    def run(self, train_list: list = None, test_list: list = None, corruption_params: dict = None,
            show_plot: bool = True) -> None:
        if corruption_params is None:
            corruption_params = {}

        if train_list is not None and len(train_list) > 0:
            self.train(t_list=train_list)

        if test_list is not None and len(test_list) > 0:
            self.test(t_list=test_list, corruption_params=corruption_params, show_plot=show_plot)
