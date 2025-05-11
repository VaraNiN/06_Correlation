import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

CACHE_EXPIRY_DAYS = 7  # Adjust as needed
CACHE_DIR = "asset_stats_cache"  # Directory to store individual files

def get_asset_stats(ticker):
    """
    Fetches historical pricing data for a given asset using yfinance, with caching to individual files.

    Args:
        ticker (str): The ticker symbol of the asset
                       (e.g., "^GSPC" for S&P 500, "URTH" for MSCI World,
                        "GC=F" for Gold, "BTC-USD" for Bitcoin, etc.).

    Returns:
        list: A list of dictionaries containing the historical pricing data, or None if an error occurs.
    """
    try:
        # Ensure the cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{ticker}_history.json")

        history = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                history = cached_data.get("history", [])
                # Convert timestamps to naive datetime objects
                last_cached_date = datetime.fromisoformat(history[-1]["timestamp"]).replace(tzinfo=None)
                if (datetime.now().replace(tzinfo=None) - last_cached_date).days < CACHE_EXPIRY_DAYS:
                    print(f"Using cached historical data for {ticker} from {cache_file}")
                    return history
            except json.JSONDecodeError:
                print(f"Error decoding cache file {cache_file}. Re-fetching.")
                os.remove(cache_file)  # Delete the corrupted file

        # Fetch historical pricing data if cache is expired or doesn't exist
        asset = yf.Ticker(ticker)

        # Check if the currency is USD
        asset_info = asset.info
        if "currency" in asset_info and asset_info["currency"] != "USD":
            raise ValueError(f"Error: The currency for {ticker} is {asset_info['currency']}, not USD.")
        

        hist_data = asset.history(period="max")  # Fetch maximum available historical data
        if not hist_data.empty:
            # Convert historical data to a list of dictionaries
            history = [
                {"timestamp": str(index), "data": row.to_dict()}
                for index, row in hist_data.iterrows()
            ]

            # Save updated history to cache
            try:
                with open(cache_file, "w") as f:
                    json.dump({"history": history}, f, indent=4)
            except Exception as e:
                print(f"Error saving historical data for {ticker} to {cache_file}: {e}")

            return history

        return None

    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return None


def load_closing_prices(tickers):
    """
    Loads the closing prices for each ticker, ensuring all tickers have data for the same dates.

    Args:
        tickers (list): A list of ticker symbols.

    Returns:
        pd.DataFrame: A DataFrame containing aligned closing prices for all tickers.
    """
    closing_prices = {}
    date_sets = {}

    # Step 1: Fetch historical data and collect dates for each ticker
    for ticker in tickers:
        history = get_asset_stats(ticker)
        if history:
            # Extract dates and store them in a set
            dates = {datetime.fromisoformat(record["timestamp"]).date() for record in history}
            date_sets[ticker] = dates

            # Store the full history temporarily
            closing_prices[ticker] = {
                datetime.fromisoformat(record["timestamp"]).date(): record["data"]["Close"]
                for record in history
            }
            print(f"Loaded {len(closing_prices[ticker])} closing prices for {ticker}.")

    # Step 2: Find common dates across all tickers
    if date_sets:
        common_dates = set.intersection(*date_sets.values())
        print(f"Found {len(common_dates)} common dates across all tickers.")

        # Step 3: Align the data for each ticker to include only common dates
        aligned_data = {
            ticker: [closing_prices[ticker][date] for date in sorted(common_dates)]
            for ticker in tickers
        }

        # Convert to a DataFrame
        df = pd.DataFrame(aligned_data, index=sorted(common_dates))
        print(f"Aligned data to {len(df)} common dates.")
        return df

    return pd.DataFrame()



def main():
    """
    Fetches and prints historical closing prices for the S&P 500, MSCI World, Gold, and Bitcoin,
    and calculates the correlation between them.
    """
    tickers = ["^GSPC", "URTH", "GC=F", "BTC-USD"]
    closing_prices_df = load_closing_prices(tickers)
    correlation_matrix = closing_prices_df.corr()

    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)



if __name__ == "__main__":
    main()