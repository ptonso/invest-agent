import pandas as pd
import numpy as np


def clean_stock_data(input_csv, output_csv=None, max_interpolation=10):
    """
    Cleans stock data, interpolates missing data, and saves the cleaned data to a CSV.
    """
    stock_data = pd.read_csv(input_csv, parse_dates=['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    # Fill in missing dates and interpolate missing values
    full_date_range = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='D')
    stock_data = stock_data.reindex(full_date_range)
    stock_data = stock_data.interpolate(method='linear', limit=max_interpolation, limit_direction='forward')
    stock_data.fillna(0, inplace=True)

    if output_csv:
        stock_data.to_csv(output_csv)
        print(f"Saved cleaned stock data to {output_csv}")
    return stock_data


def clean_ipca(input_csv, output_csv):
    """
    Converts IPCA monthly data to daily variations and saves it to a CSV file.
    """
    ipca_data = pd.read_csv(input_csv)

    # Ensure Date column is in datetime format
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

    daily_ipca_df = pd.concat(daily_ipca_list).reset_index().rename(columns={'index': 'Date', 0: 'ipca'})
    daily_ipca_df.to_csv(output_csv, index=False)
    print(f"Saved cleaned IPCA data to {output_csv}")
    return daily_ipca_df


def build_variation(input_csv, output_csv):
    """
    Builds total return variation using cleaned stock data. Only processes '_close' and '_dividends' columns.
    """
    stocks = clean_stock_data(input_csv)
    total_return_data = {}

    for column in stocks.columns:
        if "_close" in column:
            stock_name = column.replace("_close", "")
            dividend_column = f"{stock_name}_dividends"
            close_price = stocks[column].to_numpy()
            dividends = stocks[dividend_column].to_numpy() if dividend_column in stocks.columns else np.zeros(len(stocks))
            total_return_variation = np.zeros(len(close_price))

            valid_mask = (close_price[:-1] != 0) & (close_price[1:] != 0)
            total_return_variation[:-1][valid_mask] = (
                ((close_price[1:][valid_mask] + dividends[1:][valid_mask]) - close_price[:-1][valid_mask])
                / close_price[:-1][valid_mask]
            )

            zero_to_nonzero_mask = (close_price[:-1] == 0) & (close_price[1:] != 0)
            nonzero_to_zero_mask = (close_price[:-1] != 0) & (close_price[1:] == 0)
            total_return_variation[:-1][zero_to_nonzero_mask] = 0
            total_return_variation[:-1][nonzero_to_zero_mask] = 0

            total_return_data[stock_name] = total_return_variation

    total_return_df = pd.DataFrame(total_return_data, index=stocks.index)
    total_return_df.index.name = 'Date'
    total_return_df.reset_index(inplace=True)
    total_return_df.to_csv(output_csv, index=False)
    print(f"Saved total return variation to {output_csv}")


def add_selic(total_return_csv, selic_csv, output_csv):
    """
    Adds SELIC variation to total return data without IPCA adjustments.
    """
    total_return_df = pd.read_csv(total_return_csv, parse_dates=['Date'])
    selic_df = pd.read_csv(selic_csv, parse_dates=['Date'])
    
    selic_df['selic'] = selic_df['selic'].interpolate(method='linear')
    selic_df['selic'] = (1 + selic_df['selic']) ** (1 / 365) - 1  # Convert to daily rate
    selic_df.set_index('Date', inplace=True)
    
    total_return_df.set_index('Date', inplace=True)
    total_return_df = selic_df[['selic']].join(total_return_df, how='right')
    total_return_df.reset_index(inplace=True)
    total_return_df.to_csv(output_csv, index=False)
    print(f"Saved total return variation with SELIC to {output_csv}")


def compute_real_value(variation_csv, ipca_csv, output_csv):
    """
    Computes real value by adjusting stock/SELIC variation data for inflation using IPCA.
    """
    variation_df = pd.read_csv(variation_csv, parse_dates=['Date'])
    ipca_df = pd.read_csv(ipca_csv, parse_dates=['Date'])
    
    variation_df.set_index('Date', inplace=True)
    ipca_df.set_index('Date', inplace=True)
    
    combined_df = variation_df.join(ipca_df, how='left')
    
    # Compute real value for each variation column by adjusting for IPCA
    for column in variation_df.columns:
        combined_df[f'{column}_real'] = (1 + combined_df[column]) / (1 + combined_df['ipca']) - 1
    
    combined_df.reset_index(inplace=True)
    combined_df.drop(columns=['ipca'], inplace=True)
    
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved real value data to {output_csv}")


if __name__ == "__main__":
    stock_input_csv = 'data/00_raw/stocks.csv'
    selic_input_csv = 'data/00_raw/indexes.csv'
    ipca_input_csv = 'data/00_raw/indexes.csv'

    cleaned_stocks_csv = 'data/01_clean/stocks.csv'
    cleaned_ipca_csv = 'data/01_clean/ipca.csv'
    variation_csv = 'data/01_clean/total_return_var.csv'


    # Step 1: Clean stock data
    clean_stock_data(stock_input_csv, cleaned_stocks_csv)

    # Step 2: Clean IPCA data
    clean_ipca(ipca_input_csv, cleaned_ipca_csv)

    # Step 3: Build total var without IPCA
    build_variation(stock_input_csv, variation_csv)

    # Step 4: Add SELIC without IPCA
    add_selic(variation_csv, selic_input_csv, variation_csv)

    # Step 5: Compute real value
    compute_real_value(variation_csv, cleaned_ipca_csv, variation_csv)
