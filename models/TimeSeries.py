from models.TimeSeriesUtils import *


class TimeSeries:

    def __init__(self, name: str, data: np.array = None, auto_load: bool = False) -> None:
        """
        **************************************************
        * Class TimeSeries                               *
        **************************************************
        :param name: name of the time series
        :param data: data of the time series (if None, load time series from features csv files)
        :param auto_load: if True, load time series from features csv files
        Exception: data not specified
        """
        self.name = name

        if data is None and not auto_load:
            raise Exception("Data not specified and auto_load is False")

        if auto_load:
            data = TimeSeriesUtils._load_from_features_csv(
                name, TimeSeriesUtils.DATA_BASE_PATH)
        self.data = np.float32(data)
        self.anomaly_mask = np.zeros(self.data.shape)

    def set_data(self, data: np.array) -> None:
        """
        Set the data of the time series
        """
        self.data = data

    def moving_average(self, step: int) -> None:
        """
        Apply moving average to the time series
        :param step: step of the moving average
        """
        self.data = TimeSeriesUtils._moving_average(self.data, step)

    def prepare_for_window(self, type: str = "sliding", window_size: int = None, stride: int = 1) -> None:
        """
        Prepare the time series for a window
        :param type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        :param stride: stride of the window
        Exception: window type not supported
        """
        self.add_window(type=type, window_size=window_size, stride=stride)
        step = stride
        if step == 'tumbling':
            step = window_size
        self.remove_window(step=step)
        self.anomaly_mask = np.zeros(self.data.shape)

    def add_window(self, type: str = "sliding", window_size: int = None, stride: int = 1) -> None:
        """
        Add a window to the time series
        :param type: type of window to add ("sliding" | "tumbling")
        :param window_size: size of the window
        Exception: window type not supported
        """
        if window_size is not None:
            self.window_size = window_size

            if type != "sliding" and type != "tumbling":
                raise Exception("Window type not supported")

            while len(self.data) % self.window_size != 0:
                self.data = self.data[:-1]

            if type == "sliding":
                self.sliding_window(window_size, stride)
            elif type == "tumbling":
                self.tumbling_window(window_size)
        else:
            raise Exception("Window size not specified")

    def sliding_window(self, window_size: int = None, stride: int = 1) -> None:
        """
        Add a sliding window to the time series
        :param window_size: size of the window
        :param stride: stride of the window
        """
        self.data = TimeSeriesUtils._sliding_window(
            self.data, window_size, stride)

    def tumbling_window(self, window_size: int = None) -> None:
        """
        Add a tumbling window to the time series
        :param window_size: size of the window
        """
        self.data = TimeSeriesUtils._tumbling_window(
            self.data, window_size)

    def remove_window(self, step: int = 1) -> None:
        """
        Remove a window from the time series (sliding or tumbling)
        :param step: step of the window
        """
        self.data = TimeSeriesUtils._remove_window(
            self.data, n_features=len(TimeSeriesUtils.DEFAULT_COLUMNS), step=step)

    def normalize(self, normalizer) -> None:
        """
        Normalize the time series
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils._normalize(self.data, normalizer)

    def normalize_inverse(self, normalizer) -> None:
        """
        Inverse normalization of the dataset processed
        :param normalizer: normalizer to use
        """
        self.data = TimeSeriesUtils._normalize_inverse(self.data, normalizer)

    def corrupt(self, corruption: str = 'random', save_path: str = None, **kwargs) -> None:
        """
        Corrupt the time series
        :param corruption: corruption to apply
        :param save_path: path to save the corrupted time series
        :return: None
        """
        self.data, anomaly_mask = TimeSeriesUtils._corrupt(
            data=self.data, corruption_type=corruption, save_path=save_path, **kwargs)
        self.anomaly_mask = np.logical_or(self.anomaly_mask, anomaly_mask)

    def freeze_zero(self, feature_col: int = None, start: int = None, end: int = None) -> None:
        """
        Freeze the time series to zero
        :param feature_col: feature column to freeze (random if None)
        :param start: start index of the time series (random if None)
        :param end: end index of the time series (random if None)
        :return: None
        """
        self.data = TimeSeriesUtils._freeze_zero(
            self.data, feature_col, start, end)

    def freeze_last_value(self, feature_col: int = None, start: int = None, end: int = None) -> None:
        """
        Freeze the time series to its last value
        :param feature_col: feature column to freeze (random if None)
        :param start: start index of the time series (random if None)
        :param end: end index of the time series (random if None)
        :return: None
        """
        self.data = TimeSeriesUtils._freeze_last_value(
            self.data, feature_col, start, end)

    def spike(self, feature_col: int = None, point: int = None, error: int = None) -> None:
        """
        Add a spike to the time series
        :param feature_col: feature column to add the spike (random if None)
        :param point: point of the spike (random if None)
        :param error: error of the spike (random if None)
        :return: None
        """
        self.data = TimeSeriesUtils._spike(
            self.data, feature_col, point, error)

    def step(self, feature_col: int = None, start: int = None, end: int = None, error: int = None) -> None:
        """
        Add a step to the time series
        :param feature_col: feature column to add the step (random if None)
        :param start: start index of the step (random if None)
        :param end: end index of the step (random if None)
        :param error: error of the step (random if None)
        :return: None
        """
        self.data = TimeSeriesUtils._step(
            self.data, feature_col, start, end, error)

    def pca(self, model) -> None:
        """
        Apply PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils._pca(self.data, model)

    def pca_inverse(self, model) -> None:
        """
        Inverse PCA to the time series
        :param model: PCA model to use
        """
        self.data = TimeSeriesUtils._pca_inverse(self.data, model)
