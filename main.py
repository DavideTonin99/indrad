from models.TimeSeries import *
from models.TimeSeriesUtils import *
from models.Dataset import *
from utils.utils import *
from classes.MainPCA import MainPCA
from classes.MainTSAI import MainTSAI
from sklearn.preprocessing import StandardScaler

train_list = [f'trajectory_{i}' for i in range(1, 11)]
# test_list = [f'trajectory_{i}' for i in range(11, 14)]
test_list = [f'trajectory_{i}' for i in [11, 12, 14]]

params = {
    'APPLY_MOVING_AVG': True,
    'MOVING_AVG_STEP': 50,
    'WINDOW_TYPE': 'sliding',
    'WINDOW_SIZE': 4000,
    'WINDOW_STRIDE': 4000,
    'APPLY_PCA': False,
    'PCA_COMPONENTS': 25,
    'NORMALIZER_MODEL': StandardScaler(),
    'THRESHOLD_TYPE': 'quantile',
    'GAUSSIAN_MIXTURE_COMPONENTS': 10,
}

corruption_params = {
    # 'trajectory_11': {
    #     'freeze_zero': [
    #         {'start': 5000, 'end': 7000},
    #     ],
    # },
    'trajectory_12': {
        'spike': [
            {'point': 1000, 'error': 500},
            {'point': 2000, 'error': 500},
            {'point': 3000, 'error': 500},
            {'point': 4000, 'error': 500},
            {'point': 5000, 'error': 500},
        ],
    },
    # 'trajectory_14': {
    #     'step': [
    #         {'start': 1000, 'end': 3000},
    #     ],
    # },
}

main = MainPCA(params=params)
main.run(train_list=train_list, test_list=test_list, corruption_params=corruption_params)

# main = MainTSAI(params=params)
# main.run(train_list=train_list)