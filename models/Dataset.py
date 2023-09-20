import numpy as np
from models.TimeSeries import TimeSeries


class Dataset:

    def __init__(self, name: str, time_series: dict) -> None:
        """
        **************************************************
        * Class Dataset                                  *
        **************************************************
        :param name: name of the dataset (Example: train, test, nominal, anomaly, ...)
        :param time series: dictionary of time series of the dataset -> [{name: TimeSeries}, ...]
        :return: None
        """
        self.name = name
        self.time_series = time_series
        self.ts_data = TimeSeries(name=name, data=np.concatenate(
            [ts.data for ts in self.time_series.values()], axis=0))

    def get_ts(self, name: str) -> TimeSeries:
        """
        Get a time series from the dataset
        :param name: name of the time series
        :return: time series
        """
        return self.time_series[name]

    def moving_average(self, step: int, ts_name:str=None, mode:str='single') -> None:
        """
        Apply moving average to the dataset
        :param step: step of the moving average
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].moving_average(step)
                else:
                    raise Exception("Time Series not found")
            else:
                for key, value in self.time_series.items():
                    value.moving_average(step)
                    self.time_series[key] = value
            self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
                [ts.data for ts in self.time_series.values()], axis=0))
        else:
            self.ts_data.moving_average(step)
            sum_len = 0
            for key, value in self.time_series.items():
                len_ts = len(value.data)
                self.time_series[key].data = self.ts_data.data[sum_len:sum_len+len_ts]
                sum_len += len_ts
        
    def prepare_for_window(self, ts_name: str = None, type: str = "sliding", window_size: int = None, stride: int = 1):
        """
        Add a window to the trajectories of the dataset
        :param ts_name: name of the time series to add the window, if is None add the window to all the dataset
        :param type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        :param stride: stride of the window
        :return: None
        Exception: window type not supported
        """                    
        if ts_name is not None:
            if ts_name in self.time_series.keys():
                self.time_series[ts_name].prepare_for_window(type=type, window_size=window_size, stride=stride)
            else:
                raise Exception("Time Series not found")
        else:
            for ts_name in self.time_series.keys():
                self.time_series[ts_name].prepare_for_window(type=type, window_size=window_size, stride=stride)
        self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
            [ts.data for ts in self.time_series.values()], axis=0))

    def add_window(self, ts_name: str = None, type: str = "sliding", window_size: int = None, stride: int = 1) -> None:
        """
        Add a window to the trajectories of the dataset
        :param ts_name: name of the time series to add the window, if is None add the window to all the dataset
        :param type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        :param stride: stride of the window
        :return: None
        Exception: window type not supported
        """
        if window_size is not None:
            self.window_size = window_size

            if type != "sliding" and type != "tumbling":
                raise Exception("Window type not supported")

            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].add_window(type=type, window_size=window_size, stride=stride)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].add_window(type=type, window_size=window_size, stride=stride)
            self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
                [ts.data for ts in self.time_series.values()], axis=0))
        else:
            raise Exception("Window size not specified")

    def remove_window(self, ts_name: str = None, step: int = 1) -> None:
        """
        Remove a window from the dataset (sliding or tumbling)
        :param ts_name: name of the time series to remove the window, if is None remove the window to all the dataset
        :param step: step of the window
        :return: None
        """
        if ts_name is not None:
            if ts_name in self.time_series.keys():
                self.time_series[ts_name].remove_window(step=step)
            else:
                raise Exception("Time Series not found")
        else:
            for ts_name in self.time_series.keys():
                self.time_series[ts_name].remove_window(step=step)
        self.ts_data = TimeSeries(name=self.name, data=np.concatenate(
            [ts.data for ts in self.time_series.values()], axis=0))

    def normalize(self, normalizer, mode: str = 'single', ts_name: str = None) -> None:
        """
        Normalize the dataset
        :param normalizer: normalizer to use
        :param mode: mode of normalization to apply ("single" | "complete")
        :param ts_name: name of the time series to normalize, if is None normalize all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].normalize(normalizer)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].normalize(normalizer)
        else:
            self.ts_data.normalize(normalizer)
            sum_len = 0
            for ts_name, ts in self.time_series.items():
                len_ts = len(ts.data)
                self.time_series[ts_name].data = self.ts_data.data[sum_len:sum_len+len_ts]
                sum_len += len_ts

    def normalize_inverse(self, normalizer, mode: str = 'single', ts_name: str = None) -> None:
        """
        Inverse normalization of the dataset processed
        :param normalizer: normalizer to use
        :param mode: mode of normalization to apply ("single" | "complete")
        :param ts_name: name of the time series to normalize, if is None normalize all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].normalize_inverse(normalizer)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].normalize_inverse(normalizer)
        else:
            self.ts_data.normalize_inverse(normalizer)
            sum_len = 0
            for ts_name, ts in self.time_series.items():
                len_ts = len(ts.data)
                self.time_series[ts_name].data = self.ts_data.data[sum_len:sum_len+len_ts]
                sum_len += len_ts

    def corrupt(self, corruption: str = 'random', mode: str = 'single', ts_name: str = None, **kwargs) -> None:
        """
        Corrupt the dataset
        :param corruption: corruption to use
        :param mode: mode of corruption to apply ("single" | "complete")
        :param ts_name: name of the time series to corrupt, if is None corrupt all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].corrupt(corruption, **kwargs)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].corrupt(corruption, **kwargs)
        else:
            self.ts_data.corrupt(corruption, **kwargs)

    def freeze_zero(self, feature_col: int = None, start: int = None, end: int = None, mode: str = 'single', ts_name: str = None) -> None:
        """
        Freeze the dataset to zero
        :param feature_col: feature column to freeze (random if None)
        :param start: start index of the dataset (random if None)
        :param end: end index of the dataset (random if None)
        :param mode: mode of freeze to apply ("single" | "complete")
        :param ts_name: name of the time series to freeze, if is None freeze all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].freeze_zero(feature_col, start, end)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].freeze_zero(feature_col, start, end)
        else:
            self.ts_data.freeze_zero(feature_col, start, end)

    def freeze_last_value(self, feature_col: int = None, start: int = None, end: int = None, mode: str = 'single', ts_name: str = None) -> None:
        """
        Freeze the dataset to its last value
        :param feature_col: feature column to freeze (random if None)
        :param start: start index of the dataset (random if None)
        :param end: end index of the dataset (random if None)
        :param mode: mode of freeze to apply ("single" | "complete")
        :param ts_name: name of the time series to freeze, if is None freeze all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].freeze_last_value(feature_col, start, end)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].freeze_last_value(feature_col, start, end)
        else:
            self.ts_data.freeze_last_value(feature_col, start, end)

    def spike(self, feature_col: int = None, point: int = None, error: int = None, mode: str = 'single', ts_name: str = None) -> None:
        """
        Add a spike to the dataset
        :param feature_col: feature column to add the spike (random if None)
        :param point: point of the spike (random if None)
        :param error: error of the spike (random if None)
        :param mode: mode of spike to apply ("single" | "complete")
        :param ts_name: name of the time series to spike, if is None spike all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].spike(feature_col, point, error)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].spike(feature_col, point, error)
        else:
            self.ts_data.spike(feature_col, point, error)

    def step(self, feature_col: int = None, start: int = None, end: int = None, error: int = None, mode: str = 'single', ts_name: str = None) -> None:
        """
        Add a step to the dataset
        :param feature_col: feature column to add the step (random if None)
        :param start: start index of the step (random if None)
        :param end: end index of the step (random if None)
        :param error: error of the step (random if None)
        :param mode: mode of step to apply ("single" | "complete")
        :param ts_name: name of the time series to step, if is None step all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].step(feature_col, start, end, error)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].step(feature_col, start, end, error)
        else:
            self.ts_data.step(feature_col, start, end, error)

    def pca(self, model, mode: str = 'single', ts_name: str = None) -> None:
        """
        Apply PCA to the dataset
        :param model: PCA model to use
        :param mode: mode of PCA to apply ("single" | "complete")
        :param ts_name: name of the time series to apply PCA, if is None apply PCA to all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].pca(model)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].pca(model)
        else:
            self.ts_data.pca(model)
            sum_len = 0
            for ts_name, ts in self.time_series.items():
                len_ts = len(ts.data)
                self.time_series[ts_name].data = self.ts_data.data[sum_len:sum_len+len_ts]
                sum_len += len_ts

    def pca_inverse(self, model, mode: str = 'single', ts_name: str = None) -> None:
        """
        Inverse PCA to the dataset
        :param model: PCA model to use
        :param mode: mode of PCA to apply ("single" | "complete")
        :param ts_name: name of the time series to inverse PCA, if is None inverse PCA to all the dataset
        :return: None
        """
        if mode == 'single':
            if ts_name is not None:
                if ts_name in self.time_series.keys():
                    self.time_series[ts_name].pca_inverse(model)
                else:
                    raise Exception("Time Series not found")
            else:
                for ts_name in self.time_series.keys():
                    self.time_series[ts_name].pca_inverse(model)
        else:
            self.ts_data.pca_inverse(model)
            sum_len = 0
            for ts_name, ts in self.time_series.items():
                len_ts = len(ts.data)
                self.time_series[ts_name].data = self.ts_data.data[sum_len:sum_len+len_ts]
                sum_len += len_ts
