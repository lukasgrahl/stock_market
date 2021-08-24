from pprint import pprint

import pandas as pd
import numpy as np
import requests
import datetime
import os
import time

from src.decorators import get_execution_time, catch_and_log_errors


@get_execution_time
@catch_and_log_errors
def load_data(ticker_dic, col, dict_key, url_base, function, interval, slices, outputsize, datatype, adjusted,
              apikey):
    print('#######################################')
    print('Loading data...')
    df_out = pd.DataFrame()

    for item in ticker_dic.keys():
        print(item)

        if item in list(ticker_dic.keys())[::5][1:]:
            print('waiting...')
            time.sleep(62)

        if function == 'TIME_SERIES_DAILY':

            url = url_base + '?' + 'function=' + function + '&symbol=' + ticker_dic[
                item] + '&outputsize=' + outputsize + '&datatype=' + datatype + '&apikey=' + apikey

        elif function == 'TIME_SERIES_INTRADAY_EXTENDED':

            url = url_base + '?' + 'function=' + function + '&symbol=' + ticker_dic[
                item] + '&interval=' + interval + '&slice=' + slices + '&adjusted=' + adjusted + '&apikey=' + apikey

        data = pd.DataFrame(requests.get(url).json()[dict_key]).loc[col]
        data = data.rename(ticker_dic[item])

        df_out = pd.concat([df_out, data], axis=1)

    return df_out


@get_execution_time
@catch_and_log_errors
def check_data_creation_date(file_name):
    print('#######################################')
    print('Checking data creation date')

    while True:
        try:
            data_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_name)).date()
            time_delta = datetime.datetime.now().date() - data_creation_time
            time_delta = time_delta.days

        except FileNotFoundError:
            print('File did not exist')
            time_delta = 1
        break

    return time_delta


@get_execution_time
@catch_and_log_errors
def get_ticker_data(name, col, dict_key, url_base, function, interval, slices, outputsize, datatype, adjusted, apikey):
    url = url_base + '?' + 'function=' + function + '&symbol=' + name + '&outputsize=' + outputsize + '&datatype=' + datatype + '&apikey=' + apikey
    data = pd.DataFrame(requests.get(url).json()[dict_key]).loc[col]
    data.rename(name, inplace=True)

    return data


@get_execution_time
@catch_and_log_errors
def check_data_order(df_in):
    print('#######################################')
    print('Checking order of data index')
    dif_index_first = datetime.datetime.now().date() - df_in.index[0]
    dif_index_last = datetime.datetime.now().date() - df_in.index[len(df_in.index) - 1]

    if dif_index_first > dif_index_last:

        # print
        print('Data is in correct order')
        df_in = df_in

    else:
        print('Data needed ordering')
        df_in = df_in.sort_index(ascending=True)

    return df_in


@get_execution_time
@catch_and_log_errors
def get_inflow_price_dev(df_in, inflow_list, dic_weights):
    print('#######################################')
    print('Calculating stock performance')
    df_out_value = pd.DataFrame(data=df_in.index).set_index('Date')
    df_out_index = pd.DataFrame(data=df_in.index).set_index('Date')

    for item in inflow_list:
        inflow_date = datetime.datetime.strptime(item[0], '%d/%m/%Y').date()

        work_df_value = pd.concat([df_in,
                                   (df_in.loc[inflow_date:].apply(
                                       lambda x: x / x.dropna()[0]).apply(
                                       lambda x: x * dic_weights[x.name]).sum(axis=1) * item[1]).rename(
                                       f'inflow_{item[0]}')],
                                  axis=1)

        work_df_index = pd.concat([df_in,
                                   (df_in.loc[inflow_date:].apply(
                                       lambda x: x / x.dropna()[0]).apply(
                                       lambda x: x * dic_weights[x.name]).sum(axis=1)).rename(f'inflow_{item[0]}')],
                                  axis=1)

        new_col_value = pd.Series(work_df_value[f'inflow_{item[0]}'])
        new_col_index = pd.Series(work_df_index[f'inflow_{item[0]}'])

        df_out_value = pd.concat([df_out_value, new_col_value], axis=1)
        df_out_index = pd.concat([df_out_index, new_col_index], axis=1)

    return df_out_value, df_out_index


@get_execution_time
@catch_and_log_errors
def get_interval_portfolio_performance(df_in, interval, interval_length, weights):
    print('#######################################')
    print(f'Calculating {interval_length} {interval} performance')
    dic_interval = {'year': 365,
                    'month': 30,
                    'week': 7}

    df_in_interval = datetime.datetime.now().date() - datetime.timedelta(interval_length * dic_interval[interval])
    df_in_interval = df_in.loc[df_in_interval:]

    df_in_interval = df_in_interval.apply(lambda x: x / x.dropna()[0]).apply(lambda x: x * weights[x.name])

    return df_in_interval


@get_execution_time
@catch_and_log_errors
def portfolio_allocation(df_input, n_trials):
    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weight

    num_assets = len(df_input.columns)
    num_portfolios = n_trials

    cov_matrix = df_input.pct_change().apply(lambda x: np.log(1 + x)).cov()
    individual_expected_returns = df_input.resample('Y').last().pct_change().mean()

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights,
                         individual_expected_returns)  # Returns are the product of individual expected returns of asset and its
        # weight
        p_ret.append(returns)
        var = cov_matrix.mul(weights).mul(weights, axis=1).sum().sum()  # Portfolio Variance
        sd = np.sqrt(var)  # Daily standard deviation
        ann_sd = sd * np.sqrt(250)  # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    data = {'Returns': p_ret, 'Volatility': p_vol}

    for counter, symbol in enumerate(df_input.columns.tolist()):
        # print(counter, symbol)
        data[symbol + ' weight'] = [w[counter] for w in p_weights]

    portfolios = pd.DataFrame(data)
    portfolios.head()  # Dataframe of the 10000 portfolios created
    portfolios['Sharp_ratio'] = portfolios.Returns / portfolios.Volatility

    return portfolios


@get_execution_time
@catch_and_log_errors
def interpolate(df_in: pd.DataFrame,
                cols: list):
    # Function that interpolates missing values for all given columns in a df
    # args: pd df
    # returns : df with interpolated columns

    df = df_in.copy()

    for item in df.columns:
        df[item] = df[item].interpolate()

    return df


@get_execution_time
@catch_and_log_errors
def get_portfolio_allocation(df_input, n_trials, curr_weight, curr_cols):
    if (curr_cols == df_input.columns).sum() != len(df_input.columns):
        print("ATTENTION CURRENT ALLOCATION SEEMS TO BE FAULTY")
        return
    else:
        print("#### CURRENT COLUMNS MATCH DATA COLUMNS ###")

    p_ret = []  # Define an empty array for portfolio returns
    p_vol = []  # Define an empty array for portfolio volatility
    p_weights = []  # Define an empty array for asset weight

    num_assets = len(df_input.columns)
    num_portfolios = n_trials

    cov_matrix = df_input.pct_change().apply(lambda x: np.log(1 + x)).cov()
    individual_expected_returns = df_input.resample('Y').last().pct_change().mean()

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        p_weights.append(weights)

        returns = np.dot(weights,
                         individual_expected_returns)  # Returns are the product of individual expected returns of asset and its
        p_ret.append(returns)

        var = cov_matrix.mul(weights).mul(weights, axis=1).sum().sum()  # Portfolio Variance
        sd = np.sqrt(var)  # Daily standard deviation
        ann_sd = sd * np.sqrt(250)  # Annual standard deviation = volatility
        p_vol.append(ann_sd)

    curr_vol = cov_matrix.mul(curr_weight).mul(curr_weight, axis=1).sum().sum()
    curr_vol = np.sqrt(curr_vol) * np.sqrt(250)

    curr_ret = np.dot(curr_weight,
                      individual_expected_returns)

    p_weights.append(curr_weight)
    p_ret.append(curr_ret)
    p_vol.append(curr_vol)

    data = {'Returns': p_ret, 'Volatility': p_vol}

    for counter, symbol in enumerate(df_input.columns.tolist()):
        # print(counter, symbol)
        data[symbol] = [w[counter] for w in p_weights]

    portfolios = pd.DataFrame(data)
    portfolios['Sharp_ratio'] = portfolios.Returns / portfolios.Volatility

    portfolios.index = portfolios.index.astype(int)

    return portfolios


@get_execution_time
@catch_and_log_errors
def get_min_alloc_change(df_in, risk_cols, nlarg=20, curr="current allocation", nlarg_by="Sharp_ratio",
                         xlarg_col="X_LARGEST"):
    df = df_in.copy()

    df_alloc = df[df[xlarg_col] != curr].nlargest(nlarg, nlarg_by)

    curr_weights = np.array(df[df[xlarg_col] == curr][risk_cols])
    list_diff = []

    for item in df_alloc.index:
        item_weights = np.array(df_alloc.loc[item][risk_cols])
        list_diff.append((curr_weights - item_weights).sum())

    index_min_diff = pd.DataFrame(data=list_diff,
                                  index=list(df_alloc.index),
                                  columns=['diff']).nsmallest(1, "diff").index[0]

    return index_min_diff


def apply_datetime_format(x):

    x = str(x)
    try:
        x = datetime.datetime.strptime(x, "%Y-%m-%d")
        return x
    except ValueError:
        pass

    try:
        x = datetime.datetime.strptime(x, "%m.%d.%Y")
        return x
    except ValueError:
        pass

    try:
        x = datetime.datetime.strptime(x, "%m.%d.%Y %H:%M:%S")
        return x
    except ValueError:
        pass

    try:
        x = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return x
    except ValueError:
        pass

    pprint("Datetime Assignment failed")
    raise ValueError(501)
