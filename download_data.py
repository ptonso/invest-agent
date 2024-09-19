import warnings
import yfinance as yf
import pandas as pd
from bcb import sgs
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, message="The 'unit' keyword in TimedeltaIndex construction is deprecated and will be removed in a future version.")
warnings.filterwarnings("ignore", category=UserWarning, message="No timezone found, symbol may be delisted")


brazilian_stocks = [
    "ABEV3.SA",  # Ambev
    "B3SA3.SA",  # B3
    "BBAS3.SA",  # Banco do Brasil
    "BBDC3.SA",  # Bradesco
    "BBDC4.SA",  # Bradesco
    "BBSE3.SA",  # BB Seguridade
    "BPAC11.SA", # BTG Pactual
    "BRAP4.SA",  # Bradespar
    "BRFS3.SA",  # BRF
    "BRKM5.SA",  # Braskem
    "BRML3.SA",  # BR Malls
    "BTOW3.SA",  # B2W Digital
    "CCRO3.SA",  # CCR
    "CIEL3.SA",  # Cielo
    "CMIG4.SA",  # Cemig
    "CPFE3.SA",  # CPFL Energia
    "CPLE6.SA",  # Copel
    "CSAN3.SA",  # Cosan
    "CSNA3.SA",  # CSN
    "CYRE3.SA",  # Cyrela
    "ECOR3.SA",  # Ecorodovias
    "EGIE3.SA",  # Engie Brasil
    "ELET3.SA",  # Eletrobras
    "ELET6.SA",  # Eletrobras
    "EMBR3.SA",  # Embraer
    "ENBR3.SA",  # Energias do Brasil
    "ENEV3.SA",  # Eneva
    "ENGI11.SA", # Energisa
    "EQTL3.SA",  # Equatorial Energia
    "EZTC3.SA",  # EZTec
    "FLRY3.SA",  # Fleury
    "GGBR4.SA",  # Gerdau
    "GOAU4.SA",  # Metalurgica Gerdau
    "GOLL4.SA",  # Gol Linhas Aereas
    "HYPE3.SA",  # Hypera
    "IGTA3.SA",  # Iguatemi
    "IRBR3.SA",  # IRB Brasil RE
    "ITSA4.SA",  # Itausa
    "ITUB3.SA",  # Itau Unibanco
    "ITUB4.SA",  # Itau Unibanco
    "JBSS3.SA",  # JBS
    "KLBN11.SA", # Klabin
    "LAME3.SA",  # Lojas Americanas
    "LAME4.SA",  # Lojas Americanas
    "LREN3.SA",  # Lojas Renner
    "MGLU3.SA",  # Magazine Luiza
    "MRFG3.SA",  # Marfrig
    "MRVE3.SA",  # MRV
    "MULT3.SA",  # Multiplan
    "NTCO3.SA",  # Natura
    "PETR3.SA",  # Petrobras
    "PETR4.SA",  # Petrobras
    "POMO4.SA",  # Marcopolo
    "PRIO3.SA",  # PetroRio
    "QUAL3.SA",  # Qualicorp
    "RADL3.SA",  # Raia Drogasil
    "RAIL3.SA",  # Rumo
    "RENT3.SA",  # Localiza
    "SANB11.SA", # Santander Brasil
    "SBSP3.SA",  # Sabesp
    "SULA11.SA", # Sul America
    "SUZB3.SA",  # Suzano
    "TAEE11.SA", # Taesa
    "TIMP3.SA",  # TIM
    "UGPA3.SA",  # Ultrapar
    "USIM5.SA",  # Usiminas
    "VALE3.SA",  # Vale
    "VIVT3.SA",  # Telefonica Brasil
    "WEGE3.SA",  # Weg
    "YDUQ3.SA",  # Yduqs
    # Add more tickers as needed
]


def save_stock_csv(csv_path, tickers, start_date=None):
    """
    Downloads stock price and dividend data for the specified tickers and saves it to a CSV file.
    """
    combined_data = None

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        if start_date:
            price_data = stock.history(start=start_date)['Close']
            dividends_data = stock.dividends[stock.dividends.index >= start_date]
        else:
            price_data = stock.history(period='max')['Close']
            dividends_data = stock.dividends

        if price_data.empty or dividends_data.empty:
            print(f"{ticker}: No data found, symbol may be delisted or inactive.")
            continue

        price_df = price_data.reset_index().rename(columns={'Date': 'Date', 'Close': ticker})
        dividends_df = dividends_data.reset_index().rename(columns={'Date': 'Date', 'Dividends': ticker + '_dividends'})

        if combined_data is None:
            combined_data = price_df
            combined_data = combined_data.merge(dividends_df, on='Date', how='outer')
        else:
            combined_data = combined_data.merge(price_df, on='Date', how='outer')
            combined_data = combined_data.merge(dividends_df, on='Date', how='outer')

    if combined_data is not None and not combined_data.empty:
        combined_data.to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')
    else:
        raise ValueError("No valid data was fetched for any tickers. CSV file was not updated.")


series_codes = [
    # Índices de Preços
    ("selic", 432),
    ("ipca", 433),
    ("ipca_15", 7478),
    ("ipca_12m", 13522)
]


def save_indexes_csv(csv_path, series_codes, start_date=None, last_date=None):
    """
    Fetches data series from the SGS API and saves it to a CSV file.
    Rebuilds the CSV from scratch without checking existing data.
    
    Example:
    series_codes = [
        ("selic", 432),
        ("ipca", 433),
    ]
    """
    if last_date is None:
        last_date = datetime.today().strftime('%Y-%m-%d')

    combined_data = pd.DataFrame()

    for name, code in series_codes:
        try:
            new_data = sgs.get((name, code), start=start_date, end=last_date)
        except Exception as e:
            print(f"Error fetching data for {name} ({code}): {e}")
            continue

        if not new_data.empty:
            new_data.reset_index(inplace=True)
            new_data.rename(columns={'index': 'Date'}, inplace=True)

            if 'Date' not in combined_data.columns:
                combined_data = new_data[['Date', name]]
            else:
                combined_data = combined_data.merge(new_data[['Date', name]], on='Date', how='outer')
        else:
            print(f"No data fetched for {name} ({code}).")

    if not combined_data.empty:
        combined_data.to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')
    else:
        print("No data was fetched. CSV file was not updated.")



if __name__ == "__main__":

    parent = "data"
    stock_csv = f"{parent}/stocks.csv"
    index_csv = f"{parent}/indexes.csv"

    start_date = None

    print("downloading data...")

    save_indexes_csv(index_csv, series_codes, start_date, last_date=None)
    save_stock_csv(stock_csv, brazilian_stocks, start_date)

