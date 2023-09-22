from abc import abstractmethod, ABC

import numpy as np

from ad.AD import AD
from models.TimeSeries import TimeSeries
from models.Dataset import Dataset
import copy
from classes.Params import Params


class Main(ABC):
    """
    Abstract Main class
    """

    def __init__(self, params: Params) -> None:
        """
        Parameters
        ----------
        params : Params : Parameters of the model
        """
        self.params = params
        self.normalizer_model = params.NORMALIZER_MODEL
        self.ad_model = None
        self.dataset_train = None
        self.dataset_train_process = None
        self.dataset_test = None
        self.dataset_test_process = None
        self.errors = {}

    def get_dataset_name(self, prefix: str):
        """
        Get the name of the dataset
        @param prefix:
        @return:
        """
        name = prefix
        if self.params.WINDOW_TYPE is not None:
            name += f' {self.params.WINDOW_TYPE} window'
        if self.params.WINDOW_SIZE is not None:
            name += f', window size = {self.params.WINDOW_SIZE}'
        if self.params.WINDOW_STRIDE is not None:
            name += f', window stride = {self.params.WINDOW_STRIDE}'
        return name

    @staticmethod
    def load_data(t_list: list, params: Params) -> dict:
        """
        Load the data
        @param t_list: list of time series
        @param params: parameters of the model
        @return: dictionary of time series
        """
        data = {}
        window_params = {
            'window_type': params.WINDOW_TYPE,
            'window_size': params.WINDOW_SIZE,
            'window_stride': params.WINDOW_STRIDE
        }
        for t_name in t_list:
            t = TimeSeries(name=t_name, auto_load=True, **window_params)
            data[t_name] = t
        return data

    def prepare_dataset(self, name: str, data: dict, is_train: bool, corruption_params: dict = None) -> [Dataset,
                                                                                                         Dataset]:
        """
        Prepare the dataset
        @param name: name of the dataset
        @param data: data of the dataset
        @param is_train: if True, the dataset is a training dataset
        @param corruption_params: parameters of the corruption
        @return: [dataset, dataset_process]
        """
        if corruption_params is None:
            corruption_params = {}
        window_params = {
            'window_type': self.params.WINDOW_TYPE,
            'window_size': self.params.WINDOW_SIZE,
            'window_stride': self.params.WINDOW_STRIDE
        }
        dataset = Dataset(name, time_series=data, is_train=is_train, **window_params)
        if self.params.APPLY_MOVING_AVG:
            dataset.moving_average(step=self.params.MOVING_AVG_STEP)
        dataset.prepare_for_window()

        # corrupt dataset
        if len(corruption_params) > 0:
            for ts_name, p in corruption_params.items():
                for method, params_list in p.items():
                    for params in params_list:
                        if ts_name in dataset.time_series.keys():
                            dataset.corrupt(corruption=method, ts_name=ts_name, **params)

        dataset_process = copy.deepcopy(dataset)

        dataset_process.add_window()
        if is_train:
            self.normalizer_model.fit(dataset_process.ts_data.data)
        dataset_process.normalize(normalizer=self.normalizer_model)

        return [dataset, dataset_process]

    def pre_train(self, t_list: list = None) -> None:
        """
        Train the model
        @param t_list: list of training time series
        """
        if t_list is None:
            t_list = []
        data = self.load_data(t_list=t_list, params=self.params)
        name = self.get_dataset_name(prefix='train')

        self.dataset_train, self.dataset_train_process = self.prepare_dataset(name=name, data=data, is_train=True)

    def pre_test(self, t_list: list = None, corruption_params: dict = None, ad_model: AD = None) -> None:
        """
        Operations pre-test: load data, create dataset, normalize, apply sliding window
        @param t_list: list of test time series
        @param corruption_params: parameters of the corruption
        @param ad_model: anomaly detection model
        """
        if corruption_params is None:
            corruption_params = {}
        if t_list is None:
            t_list = []
        data = self.load_data(t_list=t_list, params=self.params)
        for ts_name in data.keys():
            data[ts_name].set_ad_model(ad_model)
        name = self.get_dataset_name(prefix='test')

        self.dataset_test, self.dataset_test_process = self.prepare_dataset(
            name=name, data=data, is_train=False, corruption_params=corruption_params)

    @abstractmethod
    def train(self, t_list: list) -> None:
        """
        Train the model
        @param t_list: list of training time series
        @raises NotImplementedError
        """
        pass

    @abstractmethod
    def test(self, t_list: list) -> None:
        """
        Test the model
        @param t_list: list of test time series
        """
        pass

    @abstractmethod
    def predict(self, ts: TimeSeries, ts_process: TimeSeries, threshold_params: dict = None) -> None:
        """
        Predict the time series
        @param ts: time series
        @param ts_process: processed time series
        @param threshold_params: parameters of the threshold
        @return: None
        """
        pass

    @abstractmethod
    def anomaly(self, ts_true: TimeSeries, errors: np.array, show_plot: bool = True) -> None:
        """
        Detect anomalies
        @param ts_true: true time series
        @param errors: errors
        @param show_plot: if True, show the plot
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Run execution
        """
        pass
