import json
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

CACHE_DIR = "asset_stats_cache"  # Directory to store individual files
CACHE_MAX_AGE = 1   # Maximum allowed age of cache in days
PLOT = False

ticker_label_map = {
    "^GSPC": "S&P 500",
    #"URTH": "MSCI World",
    "TLT": "20yr bonds",
    "GC=F": "Gold",
    "BTC-USD": "Bitcoin",
}



def get_asset_stats(ticker):
    """
    Fetches historical pricing data for a given asset using yfinance, with caching to individual files.

    Args:
        ticker (str): The ticker symbol of the asset.

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

                # Check the age of the youngest entry in the cached data
                if history:
                    youngest_entry_date = datetime.fromisoformat(history[-1]["timestamp"]).date()

                    # Determine the latest bank day
                    today = pd.Timestamp.now().normalize()
                    latest_bank_day = pd.bdate_range(end=today, periods=1)[0].date()

                    # Calculate the age in days relative to the latest bank day
                    age_in_days = (latest_bank_day - youngest_entry_date).days
                    if age_in_days <= CACHE_MAX_AGE:
                        return history
                    else:
                        print(f"Cached data for {ticker} is older than {CACHE_MAX_AGE} days: {age_in_days} days. Fetching fresh data.")
            except json.JSONDecodeError:
                print(f"Error decoding cache file {cache_file}. Re-fetching.")
                os.remove(cache_file)  # Delete the corrupted file

        # Fetch historical pricing data if cache is expired or doesn't exist
        print(f"Loading historical data for {ticker} from Yahoo Finance")
        asset = yf.Ticker(ticker)

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

    # Fetch historical data and collect dates for each ticker
    for ticker, label in ticker_label_map.items():
        history = get_asset_stats(ticker)
        if history:
            # Extract dates and store them in a set
            dates = {datetime.fromisoformat(record["timestamp"]).date() for record in history}
            date_sets[label] = dates

            # Store the full history temporarily
            closing_prices[label] = {
                datetime.fromisoformat(record["timestamp"]).date(): (
                    record["data"].get("Close")
                )
                for record in history
            }

    # Calculate pairwise correlation
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

                    # Calculate gains relative to the previous date
                    df = df.pct_change(fill_method=None).dropna()  # Calculate percentage change and drop NaN values

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

    # Prepare data for plotting
    relative_changes = {}
    all_dates = {}
    global_earliest_date = None

    for ticker, label in ticker_label_map.items():
        history = get_asset_stats(ticker)
        if history:
            # Determine the earliest date in the historical data
            earliest_date = min(datetime.fromisoformat(record["timestamp"]).date() for record in history)
            print(f"The oldest available data for {label} ({ticker}) is {earliest_date}")

            # Update the global earliest date
            if global_earliest_date is None or earliest_date > global_earliest_date:
                global_earliest_date = earliest_date

            # Extract closing prices
            dates = [datetime.fromisoformat(record["timestamp"]).date() for record in history]
            prices = [record["data"]["Close"] for record in history]

            # Store dates and prices for alignment
            all_dates[label] = set(dates)
            relative_changes[label] = dict(zip(dates, prices))

    # Ensure global_earliest_date is within the filtered dates
    global_earliest_date = max(global_earliest_date, min(min(dates) for dates in all_dates.values()))

    # Align data to the global earliest date and filter up to END_DATE
    aligned_changes = {}
    common_dates = sorted([date for date in set.intersection(*all_dates.values()) if date >= global_earliest_date])

    for label in ticker_label_map.values():
        aligned_changes[label] = [
            relative_changes[label][date] / relative_changes[label][global_earliest_date]
            for date in common_dates
        ]

    if PLOT:
        # Plot the data
        plt.figure(figsize=(10, 6))

        # Plot relative changes for all tickers
        for label, changes in aligned_changes.items():
            plt.plot(common_dates, changes, label=label)

        plt.title("Relative Change Over Time")
        plt.xlabel("Date")
        plt.ylabel("Relative Change (Log Scale)")
        plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    # Calculate and print daily correlation
    print("\nDaily Pairwise Correlation Matrix:")
    daily_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="D")
    print(daily_correlation)

    # Calculate and print monthly correlation
    print("\nMonthly Pairwise Correlation Matrix:")
    monthly_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="M")
    print(monthly_correlation)

    # Calculate and print yearly correlation
    print("\nYearly Pairwise Correlation Matrix:")
    year_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="Y")
    print(year_correlation)

    # Compute and print auto-correlation for each ticker
    print("\nAuto-Correlation:")
    for ticker, label in ticker_label_map.items():
        history = get_asset_stats(ticker)
        if history:
            # Extract closing prices
            dates = [datetime.fromisoformat(record["timestamp"]).date() for record in history]
            prices = [record["data"]["Close"] for record in history]

            # Create a DataFrame for resampling
            df = pd.DataFrame({"Date": dates, "Close": prices})
            df["Date"] = pd.to_datetime(df["Date"])  # Ensure the Date column is a datetime object
            df = df.set_index("Date")  # Set the Date column as the index

            # Compute auto-correlation for daily, monthly, and yearly intervals
            daily_autocorr = df["Close"].pct_change(fill_method=None).dropna().autocorr(lag=1)
            monthly_autocorr = df["Close"].resample("M").mean().pct_change(fill_method=None).dropna().autocorr(lag=1)
            yearly_autocorr = df["Close"].resample("Y").mean().pct_change(fill_method=None).dropna().autocorr(lag=1)

            print(f"{label} ({ticker}):")
            print(f"  Daily Auto-Correlation: {daily_autocorr:.4f}")
            print(f"  Monthly Auto-Correlation: {monthly_autocorr:.4f}")
            print(f"  Yearly Auto-Correlation: {yearly_autocorr:.4f}")


if __name__ == "__main__":
    main()