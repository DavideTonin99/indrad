import pandas as pd
import numpy as np
import os

from sklearn.decomposition import PCA


class TimeSeriesUtils:
    CORRUPTION_TYPES = ['freeze_zero', 'freeze_last_value', 'spike', 'step']
    DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    N_JOINTS = 7
    FEATURES = ['position', 'velocity', 'torque']

    DEFAULT_COLUMNS = [f'{key}_{i}' for key in FEATURES for i in range(7)]
    DEFAULT_COLUMNS_FEATURE = {
        'position': [f'position_{i}' for i in range(N_JOINTS)],
        'velocity': [f'velocity_{i}' for i in range(N_JOINTS)],
        'torque': [f'torque_{i}' for i in range(N_JOINTS)]
    }

    def __init__(self) -> None:
        """
        **************************************************
        * Class TimeSeriesUtils                          *
        **************************************************
        """
        pass

    @staticmethod
    def ts_moving_average(data: np.array, step: int) -> np.array:
        """
        Apply moving average to the time series
        @param data: data of the time series
        @param step: step of the moving average
        @return: np.array
        """
        result = None

        for i in range(data.shape[1]):
            if result is None:
                result = np.array(
                    [np.convolve(data[:, i], np.ones(step), 'valid') / step])
            else:
                result = np.concatenate(
                    [result, [np.convolve(data[:, i], np.ones(step), 'valid') / step]])
        return result.T

    @staticmethod
    def ts_normalize(data: np.array, normalizer) -> np.array:
        """
        Normalize the time series
        @param data: time series of the time series
        @param normalizer: normalizer to use
        """
        return normalizer.transform(data)

    @staticmethod
    def ts_normalize_inverse(data: np.array, normalizer) -> np.array:
        """
        Inverse normalization the time series
        @param data: time series of the time series
        @param normalizer: normalizer to use
        """
        return normalizer.inverse_transform(data)

    @staticmethod
    def ts_sliding_window(data: np.array, window_size: int, window_stride: int = 1) -> np.array:
        """
        Add a sliding window to the time series
        @param data: data of the time series
        @param window_size: size of the window
        @param window_stride: stride of the window
        """
        indexer = np.arange(window_size * len(TimeSeriesUtils.DEFAULT_COLUMNS))[
                  None, :] + window_stride * len(TimeSeriesUtils.DEFAULT_COLUMNS) * np.arange(
            (len(data) - (window_size - window_stride)) // window_stride)[:, None]

        return data.flatten()[indexer]

    @staticmethod
    def ts_tumbling_window(data, window_size: int = None) -> np.array:
        """
        Add a tumbling window to the time series
        @param data: data of the time series
        @param window_size: size of the window
        """
        return TimeSeriesUtils.ts_sliding_window(data, window_size, window_size)

    @staticmethod
    def ts_remove_window(data: np.array, n_features: int, step: int = 1, overlap_keep: str = 'end') -> np.array:
        """
        Remove a sliding window from the time series
        @param data: data of the time series
        @param n_features: number of features
        @param step: step of the window
        @param overlap_keep: if 'end', keep the last part of the window, if 'start', keep the first part of the window
        """
        if overlap_keep == 'end':
            result = np.concatenate([data[0].flatten(), data[1:, -(n_features * step):].flatten()]).reshape(-1,
                                                                                                            n_features)
        else:
            result = np.concatenate([data[:-1, :(n_features * step)].flatten(), data[-1].flatten()]).reshape(-1,
                                                                                                             n_features)
        return result

    @staticmethod
    def ts_corrupt(data: np.array, corruption_type: str = 'random', **kwargs) -> [np.array, np.array]:
        """
        Corrupt the time series
        @param data: data of the time series
        @param corruption_type: type of corruption to apply (random | freeze_zero | freeze_last_value | spike | step)
        @param kwargs: other parameters
        @return: [corrupted time series, mask]
        """
        if corruption_type != 'random' and corruption_type not in TimeSeriesUtils.CORRUPTION_TYPES:
            raise Exception(f"Corruption type: '{corruption_type}' not supported")

        if corruption_type == 'random':
            corruption_type = np.random.choice(
                TimeSeriesUtils.CORRUPTION_TYPES)

        return eval(f"TimeSeriesUtils.ts_{corruption_type}")(data=data, **kwargs)

    @staticmethod
    def ts_freeze_zero(data: np.array, feature_col: int = None, start: int = None, end: int = None, **kwargs) -> [
        np.array, np.array]:
        """
        Corrupt the time series with freeze zero
        @param data: data of the time series
        @param feature_col: column of the feature to corrupt (random if None)
        @param start: start of the corruption (random if None)
        @param end: end of the corruption (random if None)
        @return: [corrupted time series, mask]
        """
        if feature_col is None:
            feature_col = np.random.randint(0, len(data[0]))
        if start is None:
            start = np.random.randint(0, len(data))
        if end is None:
            end = np.random.randint(start, len(data))
        data[start:end, feature_col] = 0.0
        anomaly_mask = np.zeros(data.shape)
        anomaly_mask[start:end, feature_col] = 1.0
        return [data, anomaly_mask]

    @staticmethod
    def ts_freeze_last_value(data: np.array, feature_col: int = None, start: int = None, end: int = None, **kwargs) -> [
        np.array,
        np.array]:
        """
        Corrupt the time series with freeze last value
        @param data: data of the time series
        @param feature_col: column of the feature to corrupt (random if None)
        @param start: start of the corruption (random if None)
        @param end: end of the corruption (random if None)
        @return: [corrupted time series, mask]
        """
        if feature_col is None:
            feature_col = np.random.randint(0, len(data[0]))
        if start is None:
            start = np.random.randint(1, len(data))
        if end is None:
            end = np.random.randint(start, len(data))
        data[start:end, feature_col] = data[start - 1, feature_col]
        anomaly_mask = np.zeros(data.shape)
        anomaly_mask[start:end, :] = 1.0
        return [data, anomaly_mask]

    @staticmethod
    def ts_spike(data: np.array, feature_col: int = None, point: int = None, error: int = None, **kwargs) -> [np.array,
                                                                                                              np.array]:
        """
        Corrupt the time series with spike
        @param data: data of the time series
        @param feature_col: column of the feature to corrupt (random if None)
        @param point: point of the corruption (random if None)
        @param error: error of the corruption (random if None)
        @return: [corrupted time series, mask]
        """
        if feature_col is None:
            feature_col = np.random.randint(0, len(data[0]))
        if point is None:
            point = np.random.randint(0, len(data))
        if error is None:
            min_error = np.amax(data) * 10
            error = np.random.randint(min_error, min_error * 10)
        data[point, feature_col] += error
        anomaly_mask = np.zeros(data.shape)
        anomaly_mask[point, feature_col] = 1.0
        return [data, anomaly_mask]

    @staticmethod
    def ts_step(data: np.array, feature_col: int = None, start: int = None, end: int = None, error: int = None,
                **kwargs) -> [
        np.array, np.array]:
        """
        Corrupt the time series with step
        @param data: data of the time series
        @param feature_col: column of the feature to corrupt (random if None)
        @param start: start of the corruption (random if None)
        @param end: end of the corruption (random if None)
        @param error: error of the corruption (random if None)
        @return: [corrupted time series, mask]
        """
        if feature_col is None:
            feature_col = np.random.randint(0, len(data[0]))
        if start is None:
            start = np.random.randint(0, len(data))
        if end is None:
            end = np.random.randint(start, len(data))
        if error is None:
            min_error = np.amax(data) * 10
            error = np.random.randint(min_error, min_error * 10)
        data[start:end, feature_col] += error
        anomaly_mask = np.zeros(data.shape)
        anomaly_mask[start:end, feature_col] = 1.0
        return [data, anomaly_mask]

    @staticmethod
    def ts_pca(data: np.array, model: PCA) -> np.array:
        """
        Apply PCA to the time series
        @param data: data of the time series
        @param model: PCA model to use    
        """
        return model.transform(data)

    @staticmethod
    def ts_pca_inverse(data: np.array, model: PCA) -> np.array:
        """
        Inverse PCA to the time series
        @param data: data of the time series
        @param model: PCA model to use
        """
        return model.inverse_transform(data)

    @staticmethod
    def ts_load_from_features_csv(name: str, data_base_path: str = None) -> np.array:
        """
        Starting from position, velocity and torques csv files, it creates a dataset for each time series
        @param name: name of the time series
        @param data_base_path: path where to find the csv files
        """
        result = None

        data_base_path = data_base_path or TimeSeriesUtils.DATA_BASE_PATH

        if os.path.isdir(os.path.join(data_base_path, name)):
            data = {}
            for feature in TimeSeriesUtils.FEATURES:
                data[feature] = pd.DataFrame()
                csv_path = os.path.join(
                    data_base_path, name, f'{feature}.csv')
                if os.path.isfile(csv_path):
                    data[feature] = pd.read_csv(
                        csv_path, sep=";", header=None, names=TimeSeriesUtils.DEFAULT_COLUMNS_FEATURE[feature])
            result = pd.concat(data.values(), axis=1).to_numpy()
        else:
            raise Exception(f"Trajectory {name} not found")

        return result

    @staticmethod
    def ts_save_time_series_to_csv(names: list, output_path: str, data_base_path: str = None) -> None:
        """
        Starting from position, velocity and torques csv files, it creates a dataset for each time series and save to csv
        @param names: list of time series names to be processed
        @param output_path: path where to save the dataset
        @param data_base_path: path where to find the csv files
        """
        data_base_path = data_base_path or TimeSeriesUtils.DATA_BASE_PATH

        for name in names:
            data = TimeSeriesUtils.ts_load_from_features_csv(
                name, data_base_path)
            pd.DataFrame(data).to_csv(os.path.join(
                output_path, f'{name}.csv'), sep=";", header=False, index=False)
