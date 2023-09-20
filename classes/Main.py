from abc import abstractmethod
from models.TimeSeries import TimeSeries
from models.Dataset import Dataset
import copy
from classes.Params import Params


class Main():
    """
    Main class
    """

    def __init__(self, params: Params) -> None:
        """
        Parameters
        ----------
        params : Params : Parameters of the model
        """
        self.params = params

    def load_data(self, t_list: list) -> dict:
        """
        Load the data
        :param t_list: list of time series
        :return: dictionary of time series
        """
        data = {}
        for t_name in t_list:
            t = TimeSeries(t_name, auto_load=True)
            data[t_name] = t
        return data

    def load_dataset(self, 
                     name: str, 
                     data: dict, 
                     mode: str = 'complete', 
                     corruption_params: dict = {}
                     ) -> [Dataset, Dataset]:
        """
        Create a dataset
        :param name: name of the dataset
        :param data: data of the dataset
        :param mode: mode of the dataset
        :param corruption_params: parameters of the corruption
        :return: [dataset, dataset_process]
        """
        dataset = Dataset(name, time_series=data)
        if self.params.APPLY_MOVING_AVG:
            dataset.moving_average(
                step=self.params.MOVING_AVG_STEP, mode='single')
        dataset.prepare_for_window(
            type=self.params.WINDOW_TYPE, window_size=self.params.WINDOW_SIZE, stride=self.params.WINDOW_STRIDE)

        # corrupt dataset
        if len(corruption_params) > 0:
            for ts_name, p in corruption_params.items():
                for method, params_list in p.items():
                    for params in params_list:
                        if ts_name in dataset.time_series.keys():
                            dataset.corrupt(
                                corruption=method, mode='single', ts_name=ts_name, **params)

        dataset_process = copy.deepcopy(dataset)

        dataset_process.add_window(
            type=self.params.WINDOW_TYPE, window_size=self.params.WINDOW_SIZE, stride=self.params.WINDOW_STRIDE)
        self.normalizer_model.fit(dataset_process.ts_data.data)
        dataset_process.normalize(
            normalizer=self.normalizer_model, mode=mode)

        return [dataset, dataset_process]

    def pre_train(self, 
                  t_list: list = []
                  ) -> None:
        """
        Train the model
        :param t_list: list of training time series
        """
        data = self.load_data(t_list)
        name = 'train'
        if self.params.WINDOW_TYPE is not None:
            name += f' {self.params.WINDOW_TYPE}'
        if self.params.WINDOW_SIZE is not None:
            name += f', window ws={self.params.WINDOW_SIZE}'
        if self.params.WINDOW_STRIDE is not None:
            name += f', stride={self.params.WINDOW_STRIDE}'

        self.dataset_train, self.dataset_train_process = self.load_dataset(
            name=name, data=data, mode='complete')

    def pre_test(self, 
                 t_list: list = [], 
                 corruption_params: dict = {}
                 ) -> None:
        """
        Test the model
        :param t_list: list of test time series
        :param corruption_params: parameters of the corruption
        """
        data = self.load_data(t_list)
        name = 'test'
        if self.params.WINDOW_TYPE is not None:
            name += f' {self.params.WINDOW_TYPE}'
        if self.params.WINDOW_SIZE is not None:
            name += f', window ws={self.params.WINDOW_SIZE}'
        if self.params.WINDOW_STRIDE is not None:
            name += f', stride={self.params.WINDOW_STRIDE}'

        self.dataset_test, self.dataset_test_process = self.load_dataset(
            name=name, data=data, mode='single', corruption_params=corruption_params)

    @abstractmethod
    def prepare_for_training(self) -> None:
        """
        Prepare the model for training
        :raises NotImplementedError
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def train(self, 
              t_list: list = []
              ) -> None:
        """
        Train the model
        :param t_list: list of training time series
        :raises NotImplementedError
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def test(self, 
             t_list: list = []
             ) -> None:
        """
        Test the model
        :param t_list: list of test time series
        :raises NotImplementedError
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def evaluate(self, 
                 dataset: Dataset, 
                 mode: str
                 ) -> None:
        """
        Evaluate the model
        :param dataset: dataset to evaluate
        :param mode: mode of the evaluation (complete|single)
        :raises NotImplementedError
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def run(self) -> None:
        """
        Run the model
        :raises NotImplementedError
        """
        raise NotImplementedError("Please Implement this method")
