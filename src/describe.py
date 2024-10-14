import pandas as pd


def moving_average(stocks_data, stock, n_days):
    return stocks_data[f'{stock}_close'].rolling(window=n_days).mean()

def stock_volume(stocks_data, stock):
    return stocks_data[f'{stock}_volume']

def stock_split(stocks_data, stock):
    return stocks_data[f'{stock}_split']