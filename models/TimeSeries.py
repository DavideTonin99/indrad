from ad.AD import AD
from models.TimeSeriesUtils import *


class TimeSeries:

    def __init__(self, name: str, data: np.array = None, auto_load: bool = False, ad_model: AD = None,
                 **kwargs) -> None:
        """
        **************************************************
        * Class TimeSeries                               *
        **************************************************
        @param name: name of the time series
        @param data: data of the time series (if None, load time series from features csv files)
        @param auto_load: if True, load time series from features csv files
        @param ad_model: AD model to use
        @param kwargs: additional arguments
        @raise Exception: data not specified and auto_load is False
        """
        if data is None and not auto_load:
            raise Exception("Data not specified and auto_load is False")

        self.name = name
        self.ad_model = ad_model
        self.window_type = None
        self.window_size = None
        self.window_stride = None

        if 'window_type' in kwargs:
            self.window_type = kwargs['window_type']
        if 'window_size' in kwargs:
            self.window_size = kwargs['window_size']
        if 'window_stride' in kwargs:
            self.window_stride = kwargs['window_stride']

        # if window is tumbling, force stride to window size to avoid overlap between windows
        if self.window_type == 'tumbling':
            self.window_stride = self.window_size

        if auto_load:
            data = TimeSeriesUtils.ts_load_from_features_csv(
                name, TimeSeriesUtils.DATA_BASE_PATH)
        self.data = np.float32(data)
        self.anomaly_mask = np.zeros(self.data.shape)

    def set_data(self, data: np.array) -> None:
        """
        Set the data of the time series
        @param data: data of the time series
        """
        self.data = data

    def set_ad_model(self, ad_model: AD):
        """
        Set the AD model of the time series
        @param ad_model: AD model
        """
        self.ad_model = ad_model

    def set_window_params(self, window_type: str, window_size: int, window_stride: int = None):
        """
        Set the window parameters
        @param window_type: type of window to add ("sliding" | "tumbling")
        @param window_size: size of the window
        @param window_stride: stride of the window
        @raise Exception: window type not supported
        """
        if window_type not in ['sliding', 'tumbling']:
            raise Exception("Window type not supported")

        if window_type is not None:
            self.window_type = window_type
        if window_size is not None:
            self.window_size = window_size
        if window_type is not None:
            self.window_stride = window_stride

        if self.window_type == 'tumbling':
            self.window_stride = self.window_size

    def moving_average(self, step: int) -> None:
        """
        Apply moving average to the time series
        @param step: step of the moving average
        """
        self.data = TimeSeriesUtils.ts_moving_average(self.data, step)

    def prepare_for_window(self, window_type: str = None, window_size: int = None, window_stride: int = None) -> None:
        """
        Prepare the time series for a window
        @param window_type: type of window to add ("sliding" | "tumbling")
        @param window_size: size of the window
        @param window_stride: stride of the window
        Exception: window type not supported
        """
        if window_type is not None:
            self.set_window_params(window_type, window_size, window_stride)
        self.add_window()

        step = self.window_stride
        if step == 'tumbling':
            step = self.window_size
        self.remove_window(step=step)
        self.anomaly_mask = np.zeros(self.data.shape)

    def add_window(self, window_type: str = None, window_size: int = None, window_stride: int = None) -> None:
        """
        Add a window to the time series
        @param window_type: type of window to add ("sliding" | "tumbling")
        @param window_size: size of the window
        @param window_stride: stride of the window
        @raise Exception: window type not supported
        @raise Exception: window size not specified
        """
        if window_type is not None:
            self.set_window_params(window_type, window_size, window_stride)

        if self.window_size is None:
            raise Exception("Window size not specified")

        if self.window_type not in ['sliding', 'tumbling']:
            raise Exception("Window type not supported")

        max_len = self.window_size
        while max_len < len(self.data) - self.window_stride:
            max_len += self.window_stride

        if self.window_type == "sliding":
            self.sliding_window()
        elif self.window_type == "tumbling":
            self.tumbling_window()

    def sliding_window(self, window_size: int = None, window_stride: int = None) -> None:
        """
        Add a sliding window to the time series
        @param window_size: size of the window
        @param window_stride: stride of the window
        """
        if window_size is not None or window_stride is not None:
            self.set_window_params(window_size=window_size, window_stride=window_stride, window_type='sliding')

        self.data = TimeSeriesUtils.ts_sliding_window(
            self.data, self.window_size, self.window_stride)

    def tumbling_window(self, window_size: int = None) -> None:
        """
        Add a tumbling window to the time series
        @param window_size: size of the window
        """
        if window_size is not None:
            self.set_window_params(window_size=window_size, window_type='tumbling')

        self.data = TimeSeriesUtils.ts_tumbling_window(
            self.data, self.window_size)

    def remove_window(self, step: int = None) -> None:
        """
        Remove a window from the time series (sliding or tumbling)
        @param step: step of the window
        """
        if step is None:
            if self.window_type == 'sliding':
                step = self.window_size
            elif self.window_type == 'tumbling':
                step = self.window_stride

        self.data = TimeSeriesUtils.ts_remove_window(
            self.data, n_features=len(TimeSeriesUtils.DEFAULT_COLUMNS), step=step)

    def normalize(self, normalizer) -> None:
        """
        Normalize the time series
        @param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils.ts_normalize(self.data, normalizer)

    def normalize_inverse(self, normalizer) -> None:
        """
        Inverse normalization of the dataset processed
        @param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils.ts_normalize_inverse(self.data, normalizer)

    def corrupt(self, corruption: str = 'random', save_path: str = None, **kwargs) -> None:
        """
        Corrupt the time series
        @param corruption: corruption to apply
        @param save_path: path to save the corrupted time series
        @return: None
        """
        self.data, anomaly_mask = TimeSeriesUtils.ts_corrupt(
            data=self.data, corruption_type=corruption, save_path=save_path, **kwargs)
        self.anomaly_mask = np.logical_or(self.anomaly_mask, anomaly_mask)

    def freeze_zero(self, feature_col: int = None, start: int = None, end: int = None) -> None:
        """
        Freeze the time series to zero
        @param feature_col: feature column to freeze (random if None)
        @param start: start index of the time series (random if None)
        @param end: end index of the time series (random if None)
        @return: None
        """
        self.data = TimeSeriesUtils.ts_freeze_zero(
            self.data, feature_col, start, end)

    def freeze_last_value(self, feature_col: int = None, start: int = None, end: int = None) -> None:
        """
        Freeze the time series to its last value
        @param feature_col: feature column to freeze (random if None)
        @param start: start index of the time series (random if None)
        @param end: end index of the time series (random if None)
        @return: None
        """
        self.data = TimeSeriesUtils.ts_freeze_last_value(
            self.data, feature_col, start, end)

    def spike(self, feature_col: int = None, point: int = None, error: int = None) -> None:
        """
        Add a spike to the time series
        @param feature_col: feature column to add the spike (random if None)
        @param point: point of the spike (random if None)
        @param error: error of the spike (random if None)
        @return: None
        """
        self.data = TimeSeriesUtils.ts_spike(
            self.data, feature_col, point, error)

    def step(self, feature_col: int = None, start: int = None, end: int = None, error: int = None) -> None:
        """
        Add a step to the time series
        @param feature_col: feature column to add the step (random if None)
        @param start: start index of the step (random if None)
        @param end: end index of the step (random if None)
        @param error: error of the step (random if None)
        @return: None
        """
        self.data = TimeSeriesUtils.ts_step(
            self.data, feature_col, start, end, error)

    def pca(self, model: PCA) -> None:
        """
        Apply PCA to the time series
        @param model: PCA model to use
        """
        self.data = TimeSeriesUtils.ts_pca(self.data, model)

    def pca_inverse(self, model: PCA) -> None:
        """
        Inverse PCA to the time series
        @param model: PCA model to use
        """
        self.data = TimeSeriesUtils.ts_pca_inverse(self.data, model)

    def evaluate_ad_model(self) -> None:
        """
        Evaluate the AD model
        @return: None
        """
        self.ad_model.evaluate(ts_data=self.data)
