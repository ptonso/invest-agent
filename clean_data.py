import pandas as pd
import numpy as np


def process_stocks_to_total_return_variation(input_csv, output_csv):
    """
    Processes raw stock and dividend data into total return variation for each stock.
    
    Arguments:
    input_csv -- Path to the raw CSV file containing stock prices and dividends
    output_csv -- Path where the processed total return variation CSV will be saved
    """

    stock_data = pd.read_csv(input_csv, parse_dates=['Date'])
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.date

    stock_data.set_index('Date', inplace=True)

    full_date_range = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='D')
    stock_data = stock_data.reindex(full_date_range)

    total_return_data = {}

    for column in stock_data.columns:
        if "_dividends" not in column:
            stock_name = column
            dividend_column = f"{stock_name}_dividends"
            
            if stock_data[column].count() < 2:
                print(f"Dropping {stock_name} due to insufficient data.")
                continue

            stock_data[column] = stock_data[column].interpolate(method='akima')

            if dividend_column not in stock_data.columns:
                stock_data[dividend_column] = 0
            else:
                stock_data[dividend_column] = stock_data[dividend_column].fillna(0)

            if stock_data[column].isnull().all():
                print(f"Skipping {stock_name} due to all null stock prices.")
                continue

            total_return_variation = (
                ((stock_data[stock_name] + stock_data[dividend_column]).shift(-1) - stock_data[stock_name]) / stock_data[stock_name]
                )  # TR formula: ((P_t+1 + D_t+1) - P_t) / P_t

            # Fill any missing values with 0 for days where the stock didn't trade yet or data was unavailable
            total_return_variation = total_return_variation.fillna(0)
            total_return_data[stock_name] = total_return_variation

    total_return_df = pd.DataFrame(total_return_data, index=stock_data.index)

    total_return_df.reset_index(inplace=True)
    total_return_df.rename(columns={'index': 'Date'}, inplace=True)

    total_return_df.to_csv(output_csv, index=False)  # Set index=False so Date becomes a column
    print(f"Saved total return variation to {output_csv}")



def add_selic_to_total_return(total_return_csv, selic_csv):
    total_return_df = pd.read_csv(total_return_csv, parse_dates=['Date'])
    selic_df = pd.read_csv(selic_csv, parse_dates=['Date'])
    
    selic_df['selic'] = selic_df['selic'].interpolate(method='linear')
    selic_df.set_index('Date', inplace=True)
    
    min_date = total_return_df['Date'].min()
    max_date = total_return_df['Date'].max()
    
    selic_df = selic_df.loc[(selic_df.index >= min_date) & (selic_df.index <= max_date)]
    
    total_return_df.set_index('Date', inplace=True)
    total_return_df = selic_df[['selic']].join(total_return_df, how='right')
    
    total_return_df.reset_index(inplace=True)
    total_return_df.to_csv(total_return_csv, index=False)
    print(f"Saved total return variation with SELIC to {total_return_csv}")


def process_ipca_to_daily_variation(input_csv, output_csv):
    """
    Converts IPCA monthly data to daily variations and saves it to a CSV file.
    
    Arguments:
    input_csv -- Path to the raw CSV file containing IPCA monthly data
    output_csv -- Path where the processed daily variation CSV will be saved
    """
    ipca_data = pd.read_csv(input_csv)

    # Ensure Date column is in datetime format and IPCA column is numeric
    ipca_data['Date'] = pd.to_datetime(ipca_data['Date'])
    ipca_data['ipca'] = pd.to_numeric(ipca_data['ipca'], errors='coerce')

    daily_ipca_list = []

    for idx, row in ipca_data.iterrows():
        if not pd.isna(row['ipca']):
            month_start = row['Date']
            month_end = month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            num_days_in_month = (month_end - month_start).days + 1

            monthly_rate = row['ipca'] / 100
            daily_rate = (1 + monthly_rate) ** (1 / num_days_in_month) - 1

            daily_dates = pd.date_range(start=month_start, end=month_end, freq='D')
            daily_series = pd.Series([daily_rate] * len(daily_dates), index=daily_dates)

            daily_ipca_list.append(daily_series)

    daily_ipca_df = pd.concat(daily_ipca_list).reset_index().rename(columns={'index': 'Date', 0: 'IPCA_Daily_Variation'})
    
    daily_ipca_df.to_csv(output_csv, index=False)
    print(f"Saved IPCA daily variation to {output_csv}")

if __name__ == "__main__":
    stock_input_csv = 'data/raw/stocks.csv'
    stock_output_csv = 'data/clean/total-return-variation.csv'
    
    index_input_csv = 'data/raw/indexes.csv'
    ipca_output_csv = 'data/clean/ipca.csv'

    process_stocks_to_total_return_variation(stock_input_csv, stock_output_csv)
    process_ipca_to_daily_variation(index_input_csv, ipca_output_csv)
    add_selic_to_total_return(stock_output_csv, index_input_csv)