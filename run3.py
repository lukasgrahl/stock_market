import pandas as pd
import streamlit as st

from src.classes import DataPull, PerformanceEval, PlotlyPlots, get_risk_capital_weight, AllocationEval
from settings import inflows_etfr, inflows_trade, inflows_etfc
from settings import ticker_etfr, ticker_etfc, ticker_trade
from settings import weights, dic_time, dict_return_period, dict_interval

if __name__ == "__main__":

    # loading
    data_etfr = DataPull(ticker=ticker_etfr, request_data_name="Time Series (Daily)", file_name='data_etfr').run()

    data_etfc = DataPull(ticker=ticker_etfc, request_data_name="Time Series (Daily)", file_name="data_etfc").run()

    data_trade = DataPull(ticker=ticker_trade, request_data_name="Time Series (Daily)", file_name="data_trade").run()

    # streamlit setup
    st.set_page_config(page_title='My Apps', layout='wide')
    st.header('Portfolio Performance Overview')
    a1, a2, a3 = st.columns([1, 1, 1])
    b1, b2, b3 = st.columns([1, 2, 1])
    c1, c3 = st.columns([4, 1])
    st.header('Risk Portfolio Allocation')
    d1, d2, d3 = st.columns([1, 1, 1])
    e1, e2 = st.columns([1, 1])
    f1, f2 = st.columns([3, 1])

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
    data_trade_ind = PerformanceEval(data=data_trade,
                                     inflows=inflows_trade,
                                     port_weight=weights,
                                     dict_time=dic_time,
                                     time_period=time_period)

    data_etfr_ind = PerformanceEval(data=data_etfr,
                                    inflows=inflows_etfr,
                                    port_weight=weights,
                                    dict_time=dic_time,
                                    time_period=time_period)

    data_etfc_ind = PerformanceEval(data=data_etfc,
                                    inflows=inflows_etfc,
                                    port_weight=weights,
                                    dict_time=dic_time,
                                    time_period=time_period)

    data_all_time = pd.merge(
        (data_etfr_ind.interval_index_perf.sum(axis=1) / len(data_etfr_ind.interval_index_perf.columns)).rename('ETFR'),
        (data_etfc_ind.interval_index_perf.sum(axis=1) / len(data_etfc_ind.interval_index_perf.columns)).rename('ETFC'),
        left_index=True,
        right_index=True)

    data_all_time = pd.merge(data_all_time,
                             (data_trade_ind.interval_index_perf.sum(axis=1) / len(
                                 data_trade_ind.interval_index_perf.columns)).rename('TRADE'),
                             left_index=True,
                             right_index=True)

    overall_inflows = inflows_etfc.copy()
    overall_inflows.extend(inflows_etfr)
    overall_inflows.extend(inflows_trade)
    list1 = []

    for item in overall_inflows:
        list1.append(item[1])
    overall_inflows = sum(list1)

    # charts
    if port_chart is "all":
        b1.plotly_chart(PlotlyPlots(data=data_all_time,
                                    title="Overall Portfolio").line_plot())
        c1.write(round((data_all_time.sum(axis=1).iloc[-1] - 1), 2))
    if port_chart is "etfc":
        b1.plotly_chart(PlotlyPlots(data=data_etfc_ind.interval_index_perf,
                                    title="ETFC Portfolio").line_plot())
        c1.write(data_etfc_ind.value_perf.sum(axis=1).iloc[-1] / data_etfc_ind.total_inflows - 1)
        c1.write(data_etfc_ind.value_perf.sum(axis=1).iloc[-1] - data_etfc_ind.total_inflows)
    if port_chart is "etfr":
        b1.plotly_chart(PlotlyPlots(data=data_etfr_ind.interval_index_perf,
                                    title="ETFR Portfolio").line_plot())
        c1.write(data_etfr_ind.value_perf.sum(axis=1).iloc[-1] / data_etfr_ind.total_inflows - 1)
        c1.write(data_etfr_ind.value_perf.sum(axis=1).iloc[-1] - data_etfr_ind.total_inflows)
    if port_chart is "trade":
        b1.plotly_chart(PlotlyPlots(data=data_trade_ind.interval_index_perf,
                                    title="TRADE Portfolio").line_plot())
        c1.write(data_trade_ind.value_perf.sum(axis=1).iloc[-1] / data_trade_ind.total_inflows - 1)
        c1.write(data_trade_ind.value_perf.sum(axis=1).iloc[-1] - data_trade_ind.total_inflows)

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
        "6 month",
        "1 year",
        "2 years",
        "3 years",
        "5 years"
    ])
    sample_period = dict_interval[dic_time[sample_period][0]] * dic_time[sample_period][1]
    return_period = d3.selectbox("Return Calculation Period", [
        "year",
        "month",
        "week",
        "day"
    ])

    run_alloc = e1.button("Run Alloc")

    if run_alloc:
        # allocation
        risk_weight, data_risk = get_risk_capital_weight(dataetfc=data_etfc,
                                                         datatrade=data_trade,
                                                         inflow_etfc=inflows_etfc,
                                                         inflow_trade=inflows_trade,
                                                         weights=weights)

        data_risk_port = AllocationEval(data=data_risk,
                                        weights=risk_weight,
                                        sample_points=sample_points,
                                        sample_period=sample_period,
                                        return_period=dict_return_period[return_period])

        data_risk_port_val = data_risk_port.run()

        # charts
        f1.plotly_chart(PlotlyPlots(data=data_risk_port_val,
                                    title=f"Risk Allocation (Return Period {data_risk_port.return_period})",
                                    update={"marker_coloraxis": None},
                                    x='Volatility',
                                    y='Returns',
                                    size="X_LARGEST",
                                    color="X_LARGEST"
                                    ).scatter_plot())

        # allocation selection

    else:
        e1.write("Press Button to run allocation simulation")
