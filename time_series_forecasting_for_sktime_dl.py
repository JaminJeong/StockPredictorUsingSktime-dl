# # Regression and forecasting with sktime-dl
# [Github](https://github.com/sktime/sktime-dl)
# 
# we use sktime-dl to perform regression and forecasting on univariate time series data by deep learning.
# 
# See [sktime](https://github.com/alan-turing-institute/sktime/blob/master/examples/forecasting.ipynb) for the same forecasting performed using time series algorithms.

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from pathlib import Path
import argparse

import sys
# sys.path.append("./sktime")
sys.path.append("./sktime-dl")
sys.path.append("./sktime")

from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster

from sktime_dl.deeplearning import CNNRegressor, MLPRegressor
from sktime_dl.deeplearning import MLPRegressor, LSTMRegressor, ResNetRegressor
from sktime_dl.deeplearning import CNNRegressor, InceptionTimeRegressor, EncoderRegressor
from sktime_dl.deeplearning import TLENETRegressor, SimpleRNNRegressor, FCNRegressor

def ReadTimeSeriesDataset(train_data_path:Path, test_data_path:Path):
    train_data = None
    test_data = None
    if '.csv' in str(train_data_path):
        train_data = pd.read_csv(str(train_data_path))
    if '.csv' in str(test_data_path):
        test_data = pd.read_csv(str(test_data_path))
    return train_data, test_data


def get_path_dict():
    path_dict = {   
        "graph" : Path("./graph"),
        "model" : Path("./model"),
        "data" : Path("./data"),
    }
    
    for key, value in path_dict.items():
        if not value.exists():
            value.mkdir(parents=True)
    
    return path_dict

def get_regressor(algo_name : str, **kwargs):
    if algo_name == 'mlp':
        return MLPRegressor(**kwargs)   
    elif algo_name == 'cnn':
        return CNNRegressor(**kwargs)
    elif algo_name == 'lstm':
        return LSTMRegressor(**kwargs)
    elif algo_name == 'resnet':
        return ResNetRegressor(**kwargs)
    elif algo_name == 'inception':
        return InceptionTimeRegressor(**kwargs)
    elif algo_name == 'rnn':
        return SimpleRNNRegressor(**kwargs)
    elif algo_name == 'tlenet':
        return TLENETRegressor(**kwargs)
    elif algo_name == 'encoder':
        return EncoderRegressor(**kwargs)
    elif algo_name == 'fcn':
        return FCNRegressor(**kwargs)
    else:
        print("dont exist algorithm's name!!")
        sys.exit(1)


def get_algo():
    return ["mlp", "cnn", "lstm", "resnet", "inception", "rnn", "tlenet", "encoder", "fcn"]

def DrawGraph(model_name:str, save_image_path, train, test, predict):
    print(f"{model_name}\'s image is saved!!")
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.4))
    train.plot(ax=ax, label='train')
    test.plot(ax=ax, label='test')
    predict.plot(ax=ax, label='predict')
    plt.title(model_name)
    ax.set(ylabel='value')
    ax.set_xlabel('Date')
    plt.legend()
    plt.savefig(str(save_image_path / '{}.png'.format(model_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="StockPredictor",
                                 description="StockPredictor", add_help=True)
    parser.add_argument('-t', '--TRAIN', help='train data csv.', default='./data/Google_Stock_Price_Train.csv', required=False)
    parser.add_argument('-v', '--VALIDATION', help='test data csv.', default='./data/Google_Stock_Price_Test.csv', required=False)
    args = parser.parse_args()

    print('ReadTimeSeriesDataset() : data load!!')
    train_data, test_data = ReadTimeSeriesDataset(Path(args.TRAIN), Path(args.VALIDATION))
    algo_list = get_algo()
    path_dict = get_path_dict()
    window_length=10

    kwargs_common = {
        "model_name": 'base',
        "model_save_directory": "./model",
        "batch_size":4096,
        "verbose":1,
        # "nb_epochs":100,
    }

    print('training start!!')
    for algo_name in algo_list:
        for key, value in train_data.items():
            # Close is excepted because of object type
            if key == 'Date' or key == 'Volume' or key == 'Cloase':
                continue
        
            train = value.copy()
            kwargs_common['model_name'] = f"{key}_{algo_name}"
            regressor = get_regressor(algo_name, **kwargs_common)
            forecaster = RecursiveTimeSeriesRegressionForecaster(
                            regressor=regressor, 
                            window_length=window_length)
            test = test_data[key].copy()
            test_len = len(test)

            print(train.head())
            forecaster.fit(train)
            
            fh = [x for x in range(test_len)] # forecasting horizon
            predict = forecaster.predict(fh)
            
            trian_test = pd.concat([train, test], ignore_index=True)
            predict = pd.DataFrame(predict).set_index(trian_test[-test_len:].index)
            
            train = train.reindex(train_data['Date'])
            DrawGraph(kwargs_common['model_name'], path_dict['graph'], trian_test[-test_len*2:-test_len], trian_test[-test_len:], predict)