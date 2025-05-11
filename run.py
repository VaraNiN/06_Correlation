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
                    #print(f"Using cached historical data for {ticker} from {cache_file}")
                    return history
            except json.JSONDecodeError:
                print(f"Error decoding cache file {cache_file}. Re-fetching.")
                os.remove(cache_file)  # Delete the corrupted file

        # Fetch historical pricing data if cache is expired or doesn't exist
        print(f"Loading historical data for {ticker} from Yahoo Finance")
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
    


def calculate_pairwise_correlation(ticker_label_map, frequency="D"):
    """
    Calculates the pairwise correlation between each pair of ticker symbols using the longest possible time available
    and resampling to the specified frequency.

    Args:
        ticker_label_map (dict): A dictionary mapping ticker symbols to their labels.
        frequency (str): The frequency for resampling the data ("D" for daily, "W" for weekly, "M" for monthly).

    Returns:
        pd.DataFrame: A DataFrame containing the pairwise correlation matrix with labels as row and column names.
    """
    closing_prices = {}
    date_sets = {}

    # Step 1: Fetch historical data and collect dates for each ticker
    for ticker, label in ticker_label_map.items():
        history = get_asset_stats(ticker)
        if history:
            # Extract dates and store them in a set
            dates = {datetime.fromisoformat(record["timestamp"]).date() for record in history}
            date_sets[label] = dates

            # Store the full history temporarily
            closing_prices[label] = {
                datetime.fromisoformat(record["timestamp"]).date(): record["data"]["Close"]
                for record in history
            }
            #print(f"Loaded {len(closing_prices[label])} closing prices for {label}.")

    # Step 2: Calculate pairwise correlation
    labels = list(ticker_label_map.values())
    pairwise_correlation = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for label1 in labels:
        for label2 in labels:
            if label1 == label2:
                pairwise_correlation.loc[label1, label2] = 1.0  # Correlation with itself is always 1
            else:
                # Find common dates between the two tickers
                common_dates = date_sets[label1].intersection(date_sets[label2])
                if common_dates:
                    # Extract aligned closing prices for the common dates
                    prices1 = {date: closing_prices[label1][date] for date in common_dates}
                    prices2 = {date: closing_prices[label2][date] for date in common_dates}

                    # Create a DataFrame for the two tickers
                    df = pd.DataFrame({
                        label1: pd.Series(prices1),
                        label2: pd.Series(prices2)
                    }, index=pd.to_datetime(sorted(common_dates)))  # Convert common dates to DatetimeIndex

                    # Resample the data to the specified frequency
                    df = df.resample(frequency).mean()

                    # Calculate correlation
                    correlation = df.corr().iloc[0, 1]  # Correlation between the two columns
                    pairwise_correlation.loc[label1, label2] = correlation
                else:
                    pairwise_correlation.loc[label1, label2] = None  # No common dates

    return pairwise_correlation


def main():
    """
    Fetches and prints historical closing prices for the S&P 500, MSCI World, Gold, and Bitcoin,
    and calculates the pairwise correlation between them for daily, weekly, monthly, and 6-month data.
    """
    ticker_label_map = {
        "^GSPC": "S&P 500",
        "URTH": "MSCI World",
        "GC=F": "Gold",
        "BTC-USD": "Bitcoin"
    }

    # Calculate and print daily correlation
    print("\nDaily Pairwise Correlation Matrix:")
    daily_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="D")
    print(daily_correlation)

    # Calculate and print weekly correlation
    print("\nWeekly Pairwise Correlation Matrix:")
    weekly_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="W")
    print(weekly_correlation)

    # Calculate and print monthly correlation
    print("\nMonthly Pairwise Correlation Matrix:")
    monthly_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="ME")
    print(monthly_correlation)

    # Calculate and print 6-month correlation
    print("\n6-Month Pairwise Correlation Matrix:")
    six_month_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="6ME")
    print(six_month_correlation)

    # Calculate and print 1 year correlation
    print("\nYearly Pairwise Correlation Matrix:")
    year_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="YE")
    print(year_correlation)

if __name__ == "__main__":
    main()