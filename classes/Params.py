class Params:
    """
    Class to hold parameters for the execution
    """

    def __init__(self, params: dict) -> None:
        """
        Parameters
        ----------
        params : dict : Parameters of the execution
        """
        self.APPLY_MOVING_AVG = None
        self.MOVING_AVG_STEP = None
        # window settings
        self.WINDOW_TYPE = None
        self.WINDOW_SIZE = None
        self.WINDOW_STRIDE = None
        # normalization settings
        self.NORMALIZER_MODEL = None
        # pca settings
        self.PCA_COMPONENTS = None
        # anomaly detection settings
        self.THRESHOLD_TYPE = None
        self.GAUSSIAN_MIXTURE_COMPONENTS = None
        # quantile settings
        self.QUANTILE_LOWER_PERCENTAGE = None
        self.QUANTILE_UPPER_PERCENTAGE = None
        self.QUANTILE_MULTIPLIER = None
        # mahalanobis settings
        self.MAHALANOBIS_MULTIPLIER = None

        for key, value in params.items():
            setattr(self, key, value)
