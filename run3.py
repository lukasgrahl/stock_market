import pandas as pd

import streamlit as st

from src.classes import DataPull, PerformanceEval, PlotlyPlots, get_risk_capital_weight, AllocationEval, Portfolio
from settings import inflows_etfr, inflows_trade, inflows_etfc
from settings import ticker_etfr, ticker_etfc, ticker_trade
from settings import weights, dic_time, dict_return_period, dict_interval

if __name__ == "__main__":

    etfr = Portfolio(ticker=ticker_etfr, filename="data_etfr", weights=weights, inflows=inflows_etfr)
    etfc = Portfolio(ticker=ticker_etfc, filename="data_etfc", weights=weights, inflows=inflows_etfc)
    trade = Portfolio(ticker=ticker_trade, filename="data_trade", weights=weights, inflows=inflows_trade)

    # streamlit setup
    st.set_page_config(page_title='My Apps', layout='wide')
    st.header('Portfolio Performance Overview')
    a1, a2, a3 = st.columns([1, 1, 1])
    b1, b2, b3 = st.columns([1, 2, 1])
    c1, c3 = st.columns([1, 4])
    c11, c13 = st.columns([1, 4])
    st.header('Risk Portfolio Allocation')
    d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
    e1, e2 = st.columns([1, 1])
    e11, e22 = st.columns([1, 1])
    f1, f2 = st.columns([5, 2])

    # streamlit select
    time_period = a1.selectbox("Time Period", [
        "5 years",
        "3 years",
        "2 years",
        "1 year",
        "6 month",
        "3 month",
        "1 week"
    ])
    port_chart = a2.selectbox("Portfolio", [
        "all",
        "etfc",
        "etfr",
        "trade"
    ])

    # calc
    etfr_perf = PerformanceEval(portfolio=etfr, time_period=time_period)
    etfc_perf = PerformanceEval(portfolio=etfc, time_period=time_period)
    trade_perf = PerformanceEval(portfolio=trade, time_period=time_period)

    data_all_time = pd.merge(
        (etfr_perf.interval_index_perf.sum(axis=1) / len(etfr_perf.interval_index_perf.columns)).rename('ETFR'),
        (etfc_perf.interval_index_perf.sum(axis=1) / len(etfc_perf.interval_index_perf.columns)).rename('ETFC'),
        left_index=True,
        right_index=True)

    data_all_time = pd.merge(data_all_time,
                             (trade_perf.interval_index_perf.sum(axis=1) / len(
                                 trade_perf.interval_index_perf.columns)).rename('TRADE'),
                             left_index=True,
                             right_index=True)

    overall_inflows = inflows_etfc.copy()
    overall_inflows.extend(inflows_etfr)
    overall_inflows.extend(inflows_trade)
    list1 = []

    for item in overall_inflows:
        list1.append(item[1])
    overall_inflows = sum(list1)

    c1.write("Percentage performance")
    c11.write("Overall performance")

    # charts
    if port_chart == "all":
        b1.plotly_chart(PlotlyPlots(data=data_all_time,
                                    title="Overall Portfolio").line_plot())
        # c3.write(round((data_all_time.sum(axis=1).iloc[-1] - 1), 2))
    if port_chart == "etfc":
        b1.plotly_chart(PlotlyPlots(data=etfc_perf.interval_index_perf,
                                    title="ETFC Portfolio").line_plot())
        # c3.write(round(etfc_perf.value_perf.sum(axis=1).iloc[-1] / etfc_perf.total_inflows - 1), 2)
        # c13.write(round(etfc_perf.value_perf.sum(axis=1).iloc[-1] - etfc_perf.total_inflows), 2)
    if port_chart == "etfr":
        b1.plotly_chart(PlotlyPlots(data=etfr_perf.interval_index_perf,
                                    title="ETFR Portfolio").line_plot())
        # c3.write(round(etfr_perf.value_perf.sum(axis=1).iloc[-1] / etfr_perf.total_inflows - 1), 2)
        # c13.write(round(etfr_perf.value_perf.sum(axis=1).iloc[-1] - etfr_perf.total_inflows), 2)
    if port_chart == "trade":
        b1.plotly_chart(PlotlyPlots(data=trade_perf.interval_index_perf,
                                    title="TRADE Portfolio").line_plot())
        # c3.write(round(trade_perf.value_perf.sum(axis=1).iloc[-1] / trade_perf.total_inflows - 1), 2)
        # c13.write(round(trade_perf.value_perf.sum(axis=1).iloc[-1] - trade_perf.total_inflows), 2)

    # streamlit select
    sample_points = d1.selectbox("Sample Points", [
        500,
        1000,
        5000,
        10000,
        20000,
        40000
    ])
    sample_period = d2.selectbox("Sample Period", [
        "3 month",
        "6 month",
        "1 year",
        "2 years",
        "3 years",
        "5 years"
    ])
    sample_period = dict_interval[dic_time[sample_period][0]] * dic_time[sample_period][1]
    return_period = d3.selectbox("Return Calculation Period", [
        "day",
        "week",
        "month",
        "year"
    ])

    risk_weight, data_risk = get_risk_capital_weight(dataetfc=etfc.data,
                                                     datatrade=trade.data,
                                                     inflow_etfc=inflows_etfc,
                                                     inflow_trade=inflows_trade,
                                                     weights=weights)

    data_risk_port = AllocationEval(data=data_risk,
                                    weights=risk_weight,
                                    sample_points=sample_points,
                                    sample_period=sample_period,
                                    return_period=dict_return_period[return_period])

    data_risk_port.get_portfolio_allocation()

    xth_largest = e11.selectbox("Xth largest point", list(range(1, 10)))
    data_risk_port.get_min_alloc_change(xth_largest=xth_largest)

    # charts
    fig1 = PlotlyPlots(data=data_risk_port.data_alloc,
                       title=f"Risk Allocation (Return Period {data_risk_port.return_period})",
                       update={"marker_coloraxis": None},
                       x='Volatility',
                       y='Returns',
                       size="X_LARGEST",
                       color="X_LARGEST",
                       height=800,
                       width=1000,
                       ).scatter_plot()

    df_best_alloc = pd.DataFrame(data=data_risk_port.best_alloc.values(),
                                 index=data_risk_port.best_alloc.keys(),
                                 columns=["Best"])
    df_best_alloc["Delta"] = data_risk_port.delta_alloc.values()

    ticker_risk = ticker_trade.copy()
    ticker_risk.update(crude_oil='LOIL.L')
    ticker_risk = {ticker_risk[item] : item for item in ticker_risk.keys()}

    fig2 = PlotlyPlots(data=df_best_alloc.rename(columns=ticker_risk), title="Change in Portolio Allocation",
                       height=800,
                       width=500).bar_plot()

    f1.plotly_chart(fig1)
    f2.plotly_chart(fig2)

    # allocation selection
