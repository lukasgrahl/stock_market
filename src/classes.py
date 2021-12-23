import datetime
import os
import time
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as plx
import requests

from utils.utils import apply_datetime_format, translate_ticker_col_names
from settings import dict_url_base, DATA_DIR, dic_time, ticker_etfc, ticker_trade, inflows_trade, weights


class Portfolio(object):

    def __init__(self,
                 ticker: dict,
                 filename: str,
                 weights: dict,
                 inflows: list,
                 data: Optional[pd.DataFrame] = None,
                 data_default_index: str = "Date"):

        self.ticker = ticker
        self.filename = filename
        self.weights = {item: weights[item] for item in self.ticker.values()}
        self.inflows = inflows

        self.data_default_index = data_default_index
        self.data = data
        self.total_inflows = None

        self._get_inflow_weight()
        self._load_data()

        pass

    def _check_data_creation_date(self):
        while True:
            try:
                data_creation_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(DATA_DIR, f"{self.filename}.csv"))).date()
                self._file_creation_tdelta = (datetime.datetime.now().date() - data_creation_time).days

            except FileNotFoundError:
                self._file_creation_tdelta = 1
            break

        pass

    def _interpolate(self):

        for item in self.data.columns:
            self.data[item] = self.data[item].interpolate()

        self.data = self.data.astype(float)

    def _check_data_order(self):
        # print('Checking order of data index')
        mask_data = self.data.reset_index()
        mask_index_first = datetime.datetime.now().date() - apply_datetime_format(
            mask_data[self.data_default_index][0]).date()
        mask_index_last = datetime.datetime.now().date() - apply_datetime_format(
            mask_data[self.data_default_index].iloc[-1]).date()

        if not mask_index_first > mask_index_last:
            self.data = self.data.sort_values("Date", ascending=True)

        pass

    def _check_data_index(self):
        if self.data is not None:
            if self.data.index.name == self.data_default_index:
                self.data.reset_index(inplace=True)
                self.data[self.data_default_index] = self.data[self.data_default_index].apply(
                    lambda x: apply_datetime_format(x))
                self.data.set_index(self.data_default_index, inplace=True)

            elif self.data_default_index in self.data.columns:
                self.data[self.data_default_index] = self.data[self.data_default_index].apply(
                    lambda x: apply_datetime_format(x))
                self.data.set_index(self.data_default_index, inplace=True)

            elif "Unnamed: 0" in self.data.columns:
                self.data[self.data_default_index] = self.data["Unnamed: 0"].apply(lambda x: apply_datetime_format(x))
                self.data.set_index(self.data_default_index, inplace=True)
                self.data.drop("Unnamed: 0", axis=1, inplace=True)

            elif self.data.index.name != self.data_default_index:
                self.data.reset_index(inplace=True)
                self.data["index"] = self.data["index"].apply(lambda x: apply_datetime_format(x))
                self.data.rename(columns={"index": self.data_default_index}, inplace=True)
                self.data.set_index(self.data_default_index, inplace=True)

        pass

    def _output_sanity_check(self):
        mask = [item for item in self.ticker.values() if item not in self.data.columns]
        mask2 = (self.data.isna().astype(int)).sum() / len(self.data)

        if len(mask) > 0:
            pprint("###### ATTENTION NOT ALL DATA FROM TICKER HAS BEEN PUlLED CORRECTLY ######")
            pprint(f"MISSING COLUMNS ARE {mask}")

        if (mask2 > 0.9).sum():
            pprint("###### ATTENTION NOT ALL DATA FROM HAS BEEN PUlLED CORRECTLY ######")
            pprint(f"NAN COLUMNS ARE")
            pprint(mask2)

        pass

    def _get_inflow_weight(self):
        self.total_inflows = []
        for item in self.inflows:
            self.total_inflows.append(item[1])
        self.total_inflows = sum(self.total_inflows)
        pass

    def _load_data(self):
        self._check_data_creation_date()
        if self._file_creation_tdelta > 0:
            _ = DataPull(filename=self.filename, ticker=self.ticker)
            self.data = _.data
        elif self._file_creation_tdelta == 0:
            self.data = pd.read_csv(os.path.join(DATA_DIR, f"{self.filename}.csv"))

        self._check_data_index()
        self._check_data_order()
        self._interpolate()
        self._output_sanity_check()

        pass


class DataPull:
    """
    Class pulls data from Alpha Vantage
    """

    def __init__(self,
                 filename: str,
                 ticker: dict,
                 request_data_name: str = "Time Series (Daily)",
                 price_kind: str = "4. close",
                 url_base: dict = dict_url_base
                 ):

        """
        :param dict_url_base: dictionary that contains the base for URL Request construction
        :param ticker: dictionary that contains all portfolio_ticker names
        :param request_data_name: data name depending on request function Time Series Daily = Time Series (Daily)
        :param price_kind: stock price type : "1. open", "2. high", "3. low", "4. close", "5. volume"
        """

        # print("#######################################")
        # print("Pulling Data")

        self.filename = filename
        self.ticker = ticker
        self.price_kind = price_kind
        self.url_base = url_base
        self.request_data_name = request_data_name

        self.data = pd.DataFrame()

        self.req_url = None
        self.file_creation_time_delta = None
        self.data = None
        self.ticker_item = None

        self._load()

    def _test_api_error(self):
        try:
            mask = self.req_data["Error Message"]
            print(f"###### ERROR WHEN REQUESTING DATA: {mask} ######")
        except KeyError:
            pass

    def _test_request_freq(self):
        try:
            mask = self.req_data["Note"]
            print(f"###### WAITING 60 sec. REQUEST LIMIT IS REACHED ######")
            print(f"{mask}")
            print(f"waiting...")
            time.sleep(62)
            self.req_data = requests.get(self.req_url).json()
        except KeyError:
            pass

    def _load(self):
        print('Loading data...')
        for self.ticker_item in self.ticker.keys():

            # Time Seires data
            if self.url_base["function"] == 'TIME_SERIES_DAILY':

                dict_mask = self.url_base.copy()
                dict_mask.update(symbol=self.ticker[self.ticker_item])

                dict_mask = {item: dict_mask[item] for item in
                             ["function", "symbol", "outputsize", "datatype", "apikey"]}

                self.req_url = self.url_base["url_base"]

                for mask_key, mask_val in dict_mask.items():
                    self.req_url = self.req_url + f"&{mask_key}=" + f"{mask_val}"

            # Intraday Trade Data
            elif self.url_base["function"] == 'TIME_SERIES_INTRADAY_EXTENDED':

                dict_mask = self.url_base.copy()
                dict_mask.update(symbol=self.ticker_item)

                dict_mask = {item: self.url_base[item] for item in
                             ["function", "symbol", "interval", "slice", "adjusted", "apikey"]}

                self.req_url = self.url_base["url_base"]

                for mask_key, mask_val in dict_mask.items():
                    self.req_url = self.req_url + f"&{mask_key}=" + f"{mask_val}"

            self.req_data = requests.get(self.req_url).json()
            self._test_api_error()
            self._test_request_freq()

            self.req_data = pd.DataFrame(self.req_data[self.request_data_name]).loc[self.price_kind]
            self.req_data = self.req_data.rename(self.ticker[self.ticker_item])

            self.data = pd.concat([self.data, self.req_data], axis=1)

        self.data.to_csv(os.path.join(DATA_DIR, f'{self.filename}.csv'))
        pass


class PerformanceEval:

    def __init__(self,
                 portfolio: Portfolio,
                 time_period: str = None,
                 weighted_index: bool = False,
                 dict_time: dict = dic_time,
                 inflows: Optional[list] = None,
                 data: Optional[pd.DataFrame] = None,
                 port_weight: Optional[dict] = None, ):

        self.portfolio = portfolio
        self.data = data
        self.inflows = inflows
        self.port_weight = port_weight

        if data is None:
            self.data = self.portfolio.data
        if inflows is None:
            self.inflows = self.portfolio.inflows
        if port_weight is None:
            self.port_weight = self.portfolio.weights

        self.time_period = time_period
        self.weighted_index = weighted_index

        self.dict_time = dict_time
        self.dict_interval = {'year': 365,
                              'month': 30,
                              'week': 7}

        self.value_perf = None
        self.index_perf = None
        self.interval_index_perf = None

        if data is not None:
            self._check_sanity()
        self._get_time_interval()
        self._get_inflow_weight()

        self._get_portfolio_performance()

    def _check_sanity(self):

        try:
            self.data.set_index("Date", inplace=True)
        except KeyError:
            if self.data.index.name != "Date":
                pprint("###### ATTENTION INDEX IS NOT DATE ######")
            pass

    def _get_inflow_weight(self):

        self.total_inflows = []
        for item in self.inflows:
            self.total_inflows.append(item[1])
        self.total_inflows = sum(self.total_inflows)

    def _fillna_with_base(self):

        for col in self.index_perf.columns:
            self.index_perf[col] = self.index_perf[col].fillna(self.index_perf[col].dropna().iloc[0])

        for col in self.interval_index_perf.columns:
            self.interval_index_perf[col] = self.interval_index_perf[col].fillna(
                self.interval_index_perf[col].dropna().iloc[0])

    def _get_time_interval(self):

        if self.time_period is not None:
            self.time_period = datetime.datetime.now().date() - datetime.timedelta(
                self.dict_time[self.time_period][1] * self.dict_interval[self.dict_time[self.time_period][0]])

            self.data = self.data.loc[self.time_period:]

    def _get_portfolio_performance(self):

        self.index_perf = pd.DataFrame(data=self.data.index).set_index("Date")
        self.value_perf = pd.DataFrame(data=self.data.index).set_index("Date")

        if self.weighted_index:
            self.interval_index_perf = self.data.apply(lambda x: x / x.dropna()[0]).apply(
                lambda x: x * self.port_weight[x.name])
        else:
            self.interval_index_perf = self.data.apply(lambda x: x / x.dropna()[0])

        for item in self.inflows:
            mask_inflow_date = apply_datetime_format(item[0]).date()

            try:
                mask_value_perf = (self.data.loc[mask_inflow_date:].apply(lambda x: x / x.dropna()[0]).apply(
                    lambda x: x * self.port_weight[x.name]).sum(axis=1) * item[1]).rename(f'inflow_{item[0]}')
            except KeyError:
                pass

            if self.weighted_index:
                mask_index_perf = (self.data.loc[mask_inflow_date:].apply(lambda x: x / x.dropna()[0]).apply(
                    lambda x: x * self.port_weight[x.name]).sum(axis=1) * (item[1] / self.total_inflows)). \
                    rename(f'inflow_{item[0]}')
            else:
                mask_index_perf = (
                    self.data.loc[mask_inflow_date:].apply(lambda x: x / x.dropna()[0]).sum(axis=1)).rename(
                    f'inflow_{item[0]}')

            self.value_perf = pd.concat([self.value_perf, mask_value_perf], axis=1)
            self.index_perf = pd.concat([self.index_perf, mask_index_perf], axis=1)
        self._fillna_with_base()

        pass


class AllocationEval:

    def __init__(self,
                 portfolio: Optional[Portfolio] = None,
                 sample_points: int = 500,
                 sample_period: int = None,
                 return_period: str = "Y",
                 x_largest: int = 100,
                 data: Optional[dict] = None,
                 ticker: Optional[dict] = None,
                 weights: Optional[dict] = None):

        """
        :param sample_points: No of points used to run portfolio allocation simulations
        :param sample_period: Period of time used to run portfolio allocation simulations
        :param x_largest: No of portfolio combinations that are displayed in a different colour
        :param weights:
        """

        # print("#######################################")
        # print("Allocation Evaluation")

        self.portfolio = portfolio

        self.sample_points = sample_points
        self.sample_period = sample_period
        self.return_period = return_period
        self.x_largest = x_largest

        self.data = data
        self.weights = weights
        self.ticker = ticker

        if data is None:
            self.data = self.portfolio.data
        if weights is None:
            self.weights = self.portfolio.weights
        if ticker is None:
            self.ticker = ticker

        self.data_alloc = None
        self.best_alloc = None
        self.curr_alloc = None
        self.delta_alloc = None

        self._select_data_time_period()

    def _select_data_time_period(self):

        if self.sample_period is not None:
            mask_date = (datetime.datetime.now() - datetime.timedelta(self.sample_period)).date()
            self.data = self.data.loc[mask_date:]

    def _introduce_xlargest_to_data_alloc(self, xth_largest: int):

        self.data_alloc['X_LARGEST'] = self.data_alloc.Sharp_ratio.apply(
            lambda x: 3 if x == self.data_alloc.nlargest(xth_largest, 'Sharp_ratio').Sharp_ratio.iloc[
                -1] else 2 if x >= self.data_alloc.nlargest(self.x_largest, 'Sharp_ratio').Sharp_ratio.min() else 1)

        self.data_alloc.iloc[-1, -1] = 4
        # 1: "ordinary"
        # 2: X_largest"
        # 3: max
        # 4: current
        self.data_alloc['X_LARGEST'] = self.data_alloc['X_LARGEST'].astype("int")

    def get_min_alloc_change(self, xth_largest: int = 1):

        self._introduce_xlargest_to_data_alloc(xth_largest)

        self.best_alloc = np.array(
            self.data_alloc[self.data_alloc["X_LARGEST"] != 3].nlargest(xth_largest, "Sharp_ratio")[
                self.data.columns].iloc[-1]).ravel()
        self.curr_alloc = np.array(self.data_alloc[self.data_alloc["X_LARGEST"] == 3][self.data.columns]).ravel()

        self.delta_alloc = dict(zip(self.data.columns, self.curr_alloc - self.best_alloc))
        self.curr_alloc = dict(zip(self.data.columns, self.curr_alloc))
        self.best_alloc = dict(zip(self.data.columns, self.best_alloc))

    def get_portfolio_allocation(self):

        p_ret = []  # Define an empty array for portfolio returns
        p_vol = []  # Define an empty array for portfolio volatility
        p_weights = []  # Define an empty array for asset weight

        num_assets = len(self.data.columns)
        num_portfolios = self.sample_points

        cov_matrix = self.data.pct_change().apply(lambda x: np.log(1 + x)).cov()

        if self.sample_period is not None:
            if self.sample_period < 365:
                self.return_period = "W"

        individual_expected_returns = self.data.resample(self.return_period).last().pct_change().mean()

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

        curr_vol = cov_matrix.mul(self.weights).mul(self.weights, axis=1).sum().sum()
        curr_vol = np.sqrt(curr_vol) * np.sqrt(250)

        curr_ret = np.dot(np.array(list(self.weights.values())),
                          individual_expected_returns)

        p_weights.append(np.array(list(self.weights.values())))
        p_ret.append(curr_ret)
        p_vol.append(curr_vol)

        self.data_alloc = {'Returns': p_ret, 'Volatility': p_vol}

        for counter, symbol in enumerate(self.data.columns.tolist()):
            self.data_alloc[symbol] = [w[counter] for w in p_weights]

        self.data_alloc = pd.DataFrame(self.data_alloc)
        self.data_alloc['Sharp_ratio'] = self.data_alloc.Returns / self.data_alloc.Volatility
        self.data_alloc.index = self.data_alloc.index.astype(int)

        self.get_min_alloc_change()
        pass


class Eval:

    def __init__(self,
                 portfolio: Portfolio,
                 time_window_train: int = 3,
                 time_window_test: int = 3,
                 time_lag: int = 0,
                 ret_period: str = "D",
                 **kwargs):

        self.__dict__.update(**kwargs)
        self.kwargs = kwargs

        self.portfolio = portfolio

        self.etf_portfolio_weights = self.portfolio.weights
        self.portfolio_ticker = self.portfolio.ticker
        self.period_return_unit = ret_period
        self.portfolio._load_data()
        self.data = self.portfolio.data

        self.end_date = (datetime.datetime.now() - datetime.timedelta(time_lag)).date()
        self.mid_date = (self.end_date - datetime.timedelta(30 * time_window_test))
        self.start_date = (self.end_date - datetime.timedelta(30 * (time_window_test + time_window_train)))

        self.data_train = self.data[self.start_date: self.mid_date].copy()
        self.data_test = self.data[self.mid_date: self.end_date].copy()

        self.train_allocation_optimisation = None
        self.expected_alloc_return = None
        self.mid_end_date_delta_test = None
        self.mid_end_date_delta_train = None
        self.best_alloc_weights = None

        self._analyse()
        self._eval()
        pass

    def _eval(self):

        arr = np.linspace(1, (1 + self.expected_alloc_return * self.mid_end_date_delta_train),
                          len(self.data_test)) - self.data_test_ind.sum(axis=1).values
        arr = (arr * (np.logspace(0, 2, len(self.data_test)) / np.logspace(0, 2, len(self.data_test)).sum())).mean()
        self.error = arr.mean()

    def _analyse(self):
        mask_alloc = AllocationEval(portfolio=self.portfolio,
                                    data=self.data_train,
                                    weights=self.etf_portfolio_weights,
                                    return_period=self.period_return_unit,
                                    **self.kwargs)
        mask_alloc.get_portfolio_allocation()

        self.train_allocation_optimisation = mask_alloc.data_alloc

        self.best_alloc_weights = dict(
            zip(self.train_allocation_optimisation.sort_values("Sharp_ratio").iloc[-1][
                    list(self.portfolio_ticker.values())].index,
                self.train_allocation_optimisation.sort_values("Sharp_ratio").iloc[-1][
                    list(self.portfolio_ticker.values())].values))

        self.expected_alloc_return = self.train_allocation_optimisation.sort_values("Sharp_ratio").iloc[-1].loc[
            "Returns"]

        self.mid_end_date_delta_test = (self.data_test.index[-1] - self.data_test.index[0]).days
        self.mid_end_date_delta_train = (self.data_train.index[-1] - self.data_train.index[0]).days

        if self.period_return_unit == "W":
            self.mid_end_date_delta_test = self.mid_end_date_delta_test / 7
            self.mid_end_date_delta_train = self.mid_end_date_delta_train / 7

        if self.period_return_unit == "M":
            self.mid_end_date_delta_test = self.mid_end_date_delta_test / 30
            self.mid_end_date_delta_train = self.mid_end_date_delta_train / 30

        self.data_test_ind = self.data_test.apply(lambda x: x / x.dropna()[0] * self.best_alloc_weights[x.name])
        self.data_train_ind = self.data_train.apply(lambda x: x / x.dropna()[0] * self.best_alloc_weights[x.name])
        pass


class PlotlyPlots:

    def __init__(self,
                 data,
                 title: str,
                 width: int = 1700,
                 height: int = 800,
                 update: Optional[dict] = None,
                 **kwargs):
        """

        :param data:
        :param title:
        :param width:
        :param height:
        :param kwargs: x : col for x,
                       y : col for y,
                       color : col for colour hue
                       size : col for size hue
        """

        self.data = data
        self.title = title
        self.width = width
        self.height = height
        self.update = update

        self.fig = None

        self.__dict__.update(**kwargs)
        self.kwargs = kwargs

    def _fig_updated(self):

        if self.update is not None:
            try:
                self.fig.update_traces(**self.kwargs)
            except Exception as e:
                try:
                    self.fig.update_yaxes(**self.kwargs)
                except Exception as e:
                    pass
        pass

    def line_plot(self):
        self.fig = plx.line(data_frame=self.data,
                            title=self.title,
                            width=self.width,
                            height=self.height,
                            **self.kwargs)
        self._fig_updated()

        return self.fig

    def scatter_plot(self):
        self.fig = plx.scatter(data_frame=self.data,
                               width=self.width,
                               height=self.height,
                               title=self.title,
                               **self.kwargs)
        self._fig_updated()

        return self.fig

    def bar_plot(self):
        self.fig = plx.bar(data_frame=self.data,
                           width=self.width,
                           height=self.height,
                           title=self.title,
                           **self.kwargs)
        self._fig_updated()

        return self.fig


# Functions
def get_risk_capital_weight(dataetfc, datatrade, inflow_etfc, inflow_trade, weights):
    risk_capital = [item[1] for item in inflow_etfc]
    risk_capital.extend([item[1] for item in inflow_trade])

    etfc_share = sum([item[1] for item in inflow_etfc]) / sum(risk_capital)
    trade_share = sum([item[1] for item in inflow_trade]) / sum(risk_capital)

    risk_weight = [weights[item] * etfc_share for item in dataetfc.columns]
    risk_weight.extend([weights[item] * trade_share for item in datatrade.columns])

    risk_cols = list(dataetfc.columns)
    risk_cols.extend(datatrade.columns)

    risk_data = dataetfc.join(datatrade)

    assert (risk_data.columns == risk_cols).sum() == len(
        risk_data.columns), "###### ATTENTION RISK WEIGHTS ARE CORRUPTED ######"

    return dict(zip(risk_cols, risk_weight)), risk_data


if __name__ == "__main__":
    pass
