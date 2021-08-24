import os
from pprint import pprint

import numpy as np
import os

PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

ticker_etfr = {
    'vanguard_north_america': 'VNRT.L',
    'vanguard_emerging_markets': 'VFEM.L',
    'vanguard_europe': 'VEUR.AS',
    'vanguard_small_cap': 'WLDS.L',
    'vanguard_japan': 'VDJP.L',
    'vanguard_asia': 'VAPX.L'
}

ticker_etfc = {
    'xtracker_harvest_csi300': 'ASHR.L',
    'xtracker_msci_china': 'XCS6.DE',
    'ishares_msci_europe': 'IMV.L',
    'lyxor_nasdaq': 'NASD.L',
    'first_trust_cloud': 'FSKY.L',
    'lg_battery_value_chain': 'BATT.L',
    'xtracker_artficial_inteligence': 'XAIX.DE',
    'ishares_robotics': 'RBOD.L',
    'ishares_digital': 'DGIT.L',
    'ishares_private_equity': 'IPRV.L'
}

ticker_union = {
    'unistrategie': 'U1I4.F'

}

ticker_trade = {
    'crude_oil': 'LOIL.L'
}

# url parts
dict_url_base = {'url_base': 'https://www.alphavantage.co/query',
                 'function': 'TIME_SERIES_DAILY',
                 'interval': '60min',
                 'slices': 'year1month3',
                 'outputsize': 'full',
                 'datatype': 'json',
                 'adjusted': 'false',
                 'apikey': 'GD5O2HJGYVV89VGE'}

inflows_etfr = [
    ['12/6/2020', 500],
    ['18/6/2020', 200],
    ['5/7/2020', 55],
    ['5/8/2020', 50],
    ['5/9/2020', 50],
    ['5/10/2020', 50],
    ['5/11/2020', 50],
    ['5/12/2020', 50],
    ['17/12/2020', 100],
    ['5/1/2021', 200],
    ['5/2/2021', 50],
    ['5/3/2021', 550],
    ['17/3/2021', 300],
    ['31/3/2021', 500],
    ['5/4/2021', 50],
    ['5/5/2021', 50],
    ['5/6/2021', 50],
    ['25/6/2021', 300],
    ['5/7/2021', 50]
]

inflows_etfc = [
    ['01/05/2021', 500],
    ['01/06/2021', 300]
]

inflows_trade = [
    ['23/07/2021', 320]
]

weights = {
    'VNRT.L': 0.399,
    'VFEM.L': 0.192,
    'VEUR.AS': 0.198,
    'WLDS.L': 0.104,
    'VDJP.L': 0.065,
    'VAPX.L': 0.05,
    'ASHR.L': 0.0893,
    'XCS6.DE': 0.0193,
    'IMV.L': 0.0206,
    'NASD.L': 0.1216,
    'FSKY.L': 0.1833,
    'BATT.L': 0.1659,
    'XAIX.DE': 0.1798,
    'RBOD.L': 0.1101,
    'DGIT.L': 0.023,
    'IPRV.L': 0.0897,
    'LOIL.L': 1
}

dic_time = {
    '1 year': ['year', 1],
    '2 years': ['year', 2],
    '3 years': ['year', 3],
    '5 years': ['year', 5],
    '6 month': ['month', 6],
    '3 month': ['month', 3],
    '1 week': ['week', 1]
}

dict_interval = {'year': 365,
                 'month': 30,
                 'week': 7}

dict_return_period = {"year" : "Y",
                      "month" : "M",
                      "week" : "W",
                      "day" : "D"}
