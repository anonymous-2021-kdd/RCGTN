from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, daily_trend,add_time_in_day=True,
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape #（num_samples, num_nodes）
    time_ind = [i % trend_size for i in range(len(df))]
    time_ind = np.array(time_ind)
    print('daily_trend shape',daily_trend.shape)
    data = np.expand_dims(df, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)


    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y ,y_trend = [], [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        y_trend_t = daily_trend[time_ind[t + y_offsets],...]
        x.append(x_t)
        y.append(y_t)
        y_trend.append(y_trend_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    y_trend = np.stack(y_trend, axis=0)
    return x, y, y_trend

def calculate_daily_trend(df,train_percent,trend_size):
    df = df[0:round(len(df)*train_percent),:]  #只使用训练集的数据计算trend
    time_ind = [i % trend_size for i in range(len(df))]
    time_ind = np.array(time_ind)
    # print('time_ind',time_ind)

    trend_list = []
    for ind in range(trend_size):
        mean_timeind = np.array(df[np.where(time_ind == ind)].mean(axis=0))
        trend_list.append(mean_timeind)

    daily_trend = np.stack(trend_list,axis=0)

    return daily_trend


def generate_train_val_test(args,input_len,output_len,trend_size):


    df = np.loadtxt(args.filename, delimiter=',')

    if 'electricity' in args.filename or 'solar' in args.filename:
        print('data_rescaled')
        max_data = df.max(axis=0)
        min_data = df.min(axis=0)
        df = (df-min_data)/(max_data-min_data)

    train_percent = 0.6
    test_percent = 0.2

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-input_len+1, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, output_len+1, 1))

    #calculate the daily_trends
    daily_trend = calculate_daily_trend(df,train_percent,trend_size)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, y_trend = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        daily_trend = daily_trend,
        add_time_in_day=True,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape, ", y_trend shape", y_trend.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * test_percent)
    num_train = round(num_samples * train_percent)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train, ytrend_train = x[:num_train], y[:num_train], y_trend[:num_train]
    # val
    x_val, y_val, ytrend_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        y_trend[num_train: num_train + num_val],
    )
    # test
    x_test, y_test, ytrend_test = x[-num_test:], y[-num_test:], y_trend[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y, _ytrend = locals()["x_" + cat], locals()["y_" + cat], locals()["ytrend_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "ytrend:", _ytrend.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            ytrend=_ytrend,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    return df,daily_trend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="../data/electricity/", help="Output directory."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="../data/electricity",
        help="Raw data readings.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_len = 24
    output_len = 12
    trend_size = 7

    print("Generating training data")
    generate_train_val_test(args,input_len,output_len,trend_size)
