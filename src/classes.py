import datetime
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
# import plotly.express as plx
import requests

from utils.utils import apply_datetime_format
from settings import dict_url_base, DATA_DIR, ticker_etfc, ticker_trade, inflows_trade


class DataPull:
    """
    Class pulls data from Alpha Vantage
    """

    def __init__(self,
                 ticker: dict,
                 request_data_name: str,
                 file_name: str,
                 price_kind: str = "4. close"):

        """
        :param dict_url_base: dictionary that contains the base for URL Request construction
        :param ticker: dictionary that contains all ticker names
        :param request_data_name: data name depending on request function Time Series Daily = Time Series (Daily)
        :param price_kind: stock price type : "1. open", "2. high", "3. low", "4. close", "5. volume"
        """

        print("#######################################")
        print("Pulling Data")
        self.data = None
        self.price_kind = price_kind
        self.url_base = dict_url_base
        self.ticker = ticker
        self.ticker_item = None
        self.file_name = file_name
        self.request_data_name = request_data_name
        self.df_out = pd.DataFrame()
        self.req_url = None
        self.file_creation_time_delta = None

    def _test_api_error(self):
        try:
            mask = self.data["Error Message"]
            print(f"###### ERROR WHEN REQUESTING DATA: {mask} ######")
        except KeyError:
            pass

    def _test_request_freq(self):
        try:
            mask = self.data["Note"]
            print(f"###### WAITING 60 sec. AS REQUEST LIMIT IS REACHED ######")
            print(f"{mask}")
            print(f"waiting...")
            time.sleep(62)
            self.data = requests.get(self.req_url).json()
        except KeyError:
            pass

    def _check_data_order(self):
        print('Checking order of data index')
        mask_index_first = datetime.datetime.now().date() - self.df_out.Date[0]
        mask_index_last = datetime.datetime.now().date() - self.df_out.Date[len(self.df_out.index) - 1]

        if mask_index_first > mask_index_last:
            print('Data is in correct order')

        else:
            print('Data needed ordering')
            self.df_out = self.df_out.sort_values("Date", ascending=True)

    def _interpolate(self):

        for item in self.df_out.columns:
            self.df_out[item] = self.df_out[item].interpolate()

    def _check_data_creation_date(self):
        print('Checking data creation date')

        while True:
            try:
                data_creation_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(DATA_DIR, f"{self.file_name}.csv"))).date()
                self.file_creation_time_delta = datetime.datetime.now().date() - data_creation_time
                self.file_creation_time_delta = self.file_creation_time_delta.days

            except FileNotFoundError:
                print('###### File did not exist ######')
                self.file_creation_time_delta = 1
            break

        pass

    def _output_sanity_check(self):

        mask = [item for item in self.ticker.values() if item not in self.df_out.columns]
        mask2 = (self.df_out.isna().astype(int)).sum() / len(self.df_out)

        if len(mask) > 0:
            pprint("###### ATTENTION NOT ALL DATA FROM HAS BEEN PUlLED CORRECTLY ######")
            pprint(f"MISSING COLUMNS ARE {mask}")

        if (mask2 > 0.9).sum():
            pprint("###### ATTENTION NOT ALL DATA FROM HAS BEEN PUlLED CORRECTLY ######")
            pprint(f"NAN COLUMNS ARE")
            pprint(mask2)

    def run(self):
        print(self.file_name)
        global req_url, df_out

        self._check_data_creation_date()

        if self.file_creation_time_delta > 0:
            print('Loading data...')

            for self.ticker_item in self.ticker.keys():
                print(self.ticker_item)

                # Time Seires data
                if self.url_base["function"] == 'TIME_SERIES_DAILY':

                    dict_mask = self.url_base.copy()
                    dict_mask.update(symbol=self.ticker[self.ticker_item])

                    dict_mask = {item: dict_mask[item] for item in
                                 ["function", "symbol", "outputsize", "datatype", "apikey"]}

                    self.req_url = "https://www.alphavantage.co/query?"

                    for mask_key, mask_val in dict_mask.items():
                        self.req_url = self.req_url + f"&{mask_key}=" + f"{mask_val}"

                # Intraday Trade Data
                elif self.url_base["function"] == 'TIME_SERIES_INTRADAY_EXTENDED':

                    dict_mask = self.url_base.copy()
                    dict_mask.update(symbol=self.ticker_item)

                    dict_mask = {item: self.url_base[item] for item in
                                 ["function", "symbol", "interval", "slice", "adjusted", "apikey"]}

                    self.req_url = "https://www.alphavantage.co/query?"

                    for mask_key, mask_val in dict_mask.items():
                        self.req_url = self.req_url + f"&{mask_key}=" + f"{mask_val}"

                self.data = requests.get(self.req_url).json()
                self._test_api_error()
                self._test_request_freq()

                self.data = pd.DataFrame(self.data[self.request_data_name]).loc[self.price_kind]
                self.data = self.data.rename(self.ticker[self.ticker_item])

                self.df_out = pd.concat([self.df_out, self.data], axis=1)

            self.df_out = self.df_out.reset_index().rename(columns={'index': 'Date'})
            self.df_out.to_csv(os.path.join(DATA_DIR, f'{self.file_name}.csv'))
            self.df_out['Date'] = self.df_out["Date"].apply(lambda x: apply_datetime_format(x).date())

            self._check_data_order()
            self._interpolate()

            self.df_out.set_index("Date", inplace=True)
            self.df_out = self.df_out.astype(float)
            self._output_sanity_check()

        else:
            print(f'{self.file_name} is already up to date')
            self.df_out = pd.read_csv(os.path.join(DATA_DIR, f'{self.file_name}.csv'))
            self.df_out['Date'] = self.df_out.Date.apply(lambda x: apply_datetime_format(x).date())

            self._check_data_order()
            self._interpolate()
            self.df_out.set_index("Date", inplace=True)
            self.df_out.drop("Unnamed: 0", axis=1, inplace=True)

            self._output_sanity_check()

        return self.df_out


class PerformanceEval:

    def __init__(self,
                 data: pd.DataFrame,
                 inflows: list,
                 port_weight: dict,
                 dict_time: dict,
                 weighted_index: bool = False,
                 time_period=None):
        self.data = data
        self.time_period = time_period
        self.weighted_index = weighted_index
        self.inflows = inflows
        self.port_weight = port_weight
        self.dict_time = dict_time
        self.dict_interval = {'year': 365,
                              'month': 30,
                              'week': 7}

        self.value_perf = None
        self.index_perf = None
        self.interval_index_perf = None

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
                 data: pd.DataFrame,
                 weights: dict,
                 sample_points: int = 500,
                 sample_period: int = None,
                 return_period: str = "Y",
                 x_largest: int = 100
                 ):

        """
        :param sample_points: No of points used to run portfolio allocation simulations
        :param sample_period: Period of time used to run portfolio allocation simulations
        :param x_largest: No of portfolio combinations that are displayed in a different colour
        :param weights:
        """

        print("#######################################")
        print("Allocation Evaluation")

        self.sample_points = sample_points
        self.sample_period = sample_period
        self.return_period = return_period
        self.x_largest = x_largest

        self.data = data

        self.weights = [weights[item] for item in self.data.columns]
        self.data_alloc = None
        self.best_alloc = None
        self.curr_weight = None
        self.delta_weight = None

    def _select_data_time_period(self):

        self.data.reset_index(inplace=True)
        self.data["Date"] = self.data.Date.apply(lambda x: apply_datetime_format(x))
        self.data.set_index("Date", inplace=True)

        if self.sample_period is not None:
            mask_date = (datetime.datetime.now() - datetime.timedelta(self.sample_period)).date()
            self.data = self.data.loc[mask_date:]

    def _introduce_xlargest_to_data_alloc(self):

        self.data_alloc['X_LARGEST'] = self.data_alloc.Sharp_ratio.apply(lambda x:
                                                                         2 if x >= self.data_alloc.nlargest(
                                                                             self.x_largest + 1,
                                                                             'Sharp_ratio').Sharp_ratio.min()
                                                                         else 1)
        self.data_alloc.iloc[-1, -1] = 3
        # self.data_alloc["X_LARGEST"].replace({0: "ordinary",
        #                                       1: f"{self.x_largest} largest",
        #                                       2: "current allocation"}, inplace=True)
        self.data_alloc['X_LARGEST'] = self.data_alloc['X_LARGEST'].astype("int")

    def _get_min_alloc_change(self):

        self.best_alloc = np.array(
            self.data_alloc[self.data_alloc["X_LARGEST"] != 2].nlargest(1, "Sharp_ratio")[
                self.data.columns])
        self.curr_weight = np.array(
            self.data_alloc[self.data_alloc["X_LARGEST"] == 2][self.data.columns])
        self.delta_weight = self.curr_weight - self.best_alloc

    def _get_portfolio_allocation(self):

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

        curr_ret = np.dot(self.weights,
                          individual_expected_returns)

        p_weights.append(self.weights)
        p_ret.append(curr_ret)
        p_vol.append(curr_vol)

        self.data_alloc = {'Returns': p_ret, 'Volatility': p_vol}

        for counter, symbol in enumerate(self.data.columns.tolist()):
            # print(counter, symbol)
            self.data_alloc[symbol] = [w[counter] for w in p_weights]

        self.data_alloc = pd.DataFrame(self.data_alloc)
        self.data_alloc['Sharp_ratio'] = self.data_alloc.Returns / self.data_alloc.Volatility
        self.data_alloc.index = self.data_alloc.index.astype(int)

        self._introduce_xlargest_to_data_alloc()

        pass

    def run(self):

        self._select_data_time_period()
        self._get_portfolio_allocation()
        self._get_min_alloc_change()

        return self.data_alloc


class AllocationSelect:

    def __init__(self):
        pass

    def run(self):
        pass


class Eval:

    def __init__(self,
                 time_lag: int,
                 time_window: int,
                 weights: dict,
                 filename: str,
                 ticker: list):

        self.weights = weights
        self.x0_month = (datetime.datetime.now() - datetime.timedelta(time_lag)).date()
        self.x1_month = ((datetime.datetime.now() - datetime.timedelta(time_lag)) - datetime.timedelta(
            30 * time_window / 2)).date()
        self.x2_month = ((datetime.datetime.now() - datetime.timedelta(time_lag)) - datetime.timedelta(
            30 * time_window)).date()

        self.filename = filename
        self.data = DataPull(ticker=ticker,
                             request_data_name="Time Series (Daily)",
                             file_name=self.filename).run().reset_index()

        self.sugg_weight = None
        self.ret = None

        self._run()
        pass

    def _run(self):
        self.data["Date"] = self.data.Date.apply(lambda x: apply_datetime_format(x))
        self.data.set_index("Date", inplace=True)

        x1 = self.data[self.x2_month: self.x1_month].copy()
        x2 = self.data[self.x1_month: self.x0_month].copy()

        x1 = AllocationEval(x1,
                            weights=self.weights,
                            return_period="D").run()

        self.sugg_weight = dict(zip(x1.sort_values("Sharp_ratio").iloc[-1][list(ticker_etfc.values())].index,
                                    x1.sort_values("Sharp_ratio").iloc[-1][list(ticker_etfc.values())].values))
        self.ret = (x2.resample("D").last().pct_change().mean() * list(self.sugg_weight.values())).sum()

        pass


# class PlotlyPlots:
#
#     def __init__(self,
#                  data,
#                  title,
#                  width=1700,
#                  height=800,
#                  update=None,
#                  **kwargs):
#         """
#
#         :param data:
#         :param title:
#         :param width:
#         :param height:
#         :param kwargs: x : col for x,
#                        y : col for y,
#                        color : col for colour hue
#                        size : col for size hue
#         """
#
#         self.data = data
#         self.title = title
#         self.width = width
#         self.height = height
#         self.update = update
#
#         self.fig = None
#
#         self.__dict__.update(**kwargs)
#         self.kwargs = kwargs
#
#     def _fig_updated(self):
#
#         if self.update is not None:
#             self.fig.update_traces(self.update)
#         else:
#             pass
#
#     def line_plot(self):
#         self.fig = plx.line(data_frame=self.data,
#                             title=self.title,
#                             width=self.width,
#                             height=self.height,
#                             **self.kwargs)
#         self._fig_updated()
#
#         return self.fig
#
#     def scatter_plot(self):
#         self.fig = plx.scatter(data_frame=self.data,
#                                width=self.width,
#                                height=self.height,
#                                title=self.title,
#                                **self.kwargs)
#         self._fig_updated()
#
#         return self.fig
#
#     def bar_plot(self):
#         self.fig = plx.bar(data_frame=self.data,
#                            width=self.width,
#                            height=self.height,
#                            title=self.title,
#                            **self.kwargs)
#         self._fig_updated()
#
#         return self.fig


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

    if not (risk_data.columns == risk_cols).sum() == len(risk_data.columns):
        pprint("###### ATTENTION RISK WEIGHTS ARE CORRUPTED ######")

    return dict(zip(risk_cols, risk_weight)), risk_data


if __name__ == "__main__":
    from settings import dict_url_base, weights

    x_month = 6
    x1_month_past = datetime.datetime.now() - datetime.timedelta(30 * x_month)
    x2_month_past = datetime.datetime.now() - datetime.timedelta(30 * (x_month / 2))

    data_etfc = DataPull(ticker=ticker_etfc,
                         request_data_name="Time Series (Daily)",
                         file_name='data_etfc').run().reset_index()

    data_etfc["Date"] = data_etfc.Date.apply(lambda x: apply_datetime_format(x))
    data_etfc.set_index("Date", inplace=True)

    data_etfc = data_etfc[x1_month_past: x2_month_past]

    x = AllocationEval(data=data_etfc,
                       return_period="D",
                       weights=weights).run()

    print(x)
