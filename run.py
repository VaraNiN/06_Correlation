import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

CACHE_EXPIRY_DAYS = 7  # Adjust as needed
CACHE_DIR = "asset_stats_cache"  # Directory to store individual files
SUBTRACT_INFLATION_FLAG = True  # Set to True to subtract inflation from returns
END_DATE = 2024  # End date for inflation data


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
                # Convert timestamps to naive datetime objects
                last_cached_date = datetime.fromisoformat(history[-1]["timestamp"]).replace(tzinfo=None)
                if (datetime.now().replace(tzinfo=None) - last_cached_date).days < CACHE_EXPIRY_DAYS:
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
            # Filter data up to END_DATE
            hist_data = hist_data[hist_data.index <= f"{END_DATE}-12-31"]

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


def fetch_yearly_inflation():   #Source; https://www.usinflationcalculator.com/inflation/historical-inflation-rates/
    """
    Loads yearly inflation data from the inflation.csv file.

    Returns:
        pd.DataFrame: A DataFrame containing the yearly inflation data.
    """
    CSV_FILE = "inflation.csv"  #Source; https://www.usinflationcalculator.com/inflation/historical-inflation-rates/

    try:
        # Load data from inflation.csv
        print("Loading inflation data from inflation.csv...")
        if not os.path.exists(CSV_FILE):
            raise FileNotFoundError(f"Inflation data file not found: {CSV_FILE}")

        # Read the CSV file and extract the Year and Ave columns
        inflation_df = pd.read_csv(CSV_FILE, usecols=["Year", "Ave"])
        inflation_df.rename(columns={"Ave": "Inflation Rate"}, inplace=True)
        inflation_df["Year"] = pd.to_datetime(inflation_df["Year"], format="%Y")  # Convert Year to datetime
        inflation_df["Inflation Rate"] = pd.to_numeric(inflation_df["Inflation Rate"], errors="coerce")  # Ensure numeric type
        inflation_df.set_index("Year", inplace=True)

        # Filter data up to END_DATE
        inflation_df = inflation_df[inflation_df.index.year <= END_DATE]

        return inflation_df

    except Exception as e:
        print(f"Error loading inflation data: {e}")
        return None


def precompute_inflation_factors(inflation_data, start_date):
    """
    Precomputes cumulative inflation factors for each date since the start date.

    Args:
        inflation_data (pd.DataFrame): A DataFrame containing yearly inflation rates with the year as the index.
        start_date (datetime): The start date for inflation adjustment.

    Returns:
        dict: A dictionary mapping each date to its cumulative inflation factor.
    """
    cumulative_factors = {}
    cumulative_factor = 1.0

    # Ensure start_date is a datetime object
    start_date = datetime.combine(start_date, datetime.min.time())

    # Iterate over each year and compute cumulative inflation factors
    for year in range(start_date.year, min(END_DATE, inflation_data.index.max().year) + 1):
        if year in inflation_data.index.year:
            inflation_rate = inflation_data.loc[inflation_data.index[inflation_data.index.year == year], "Inflation Rate"].values[0] / 100
            days_in_year = 365  # Approximation for daily compounding
            daily_inflation_factor = (1 + inflation_rate) ** (1 / days_in_year)

            # Compute inflation factors for each day in the year
            for day in range(1, days_in_year + 1):
                current_date = datetime(year, 1, 1) + pd.Timedelta(days=day - 1)
                if current_date >= start_date:
                    cumulative_factor *= daily_inflation_factor
                    cumulative_factors[current_date.date()] = cumulative_factor
        else:
            raise ValueError(f"Year {year} not found in inflation data.")

    return cumulative_factors

def adjust_for_inflation(asset_data, inflation_factors):
    """
    Adjusts the daily asset prices for inflation using precomputed inflation factors.

    Args:
        asset_data (list): A list of dictionaries containing historical pricing data from get_asset_stats.
        inflation_factors (dict): A dictionary mapping dates to cumulative inflation factors.

    Returns:
        list: A list of dictionaries with inflation-adjusted prices.
    """
    if not SUBTRACT_INFLATION_FLAG or not inflation_factors:
        return asset_data  # Return unadjusted data if the flag is False or inflation factors are missing

    adjusted_data = []

    for record in asset_data:
        try:
            # Parse the date
            date = datetime.fromisoformat(record["timestamp"]).date()

            # Adjust the closing price for cumulative inflation
            if date in inflation_factors:
                adjusted_close = record["data"]["Close"] / inflation_factors[date]
                record["data"]["Inflation Adjusted Close"] = adjusted_close
            else:
                record["data"]["Inflation Adjusted Close"] = record["data"]["Close"]

            adjusted_data.append(record)
        except Exception as e:
            print(f"Error adjusting for inflation on {record['timestamp']}: {e}")
            continue

    return adjusted_data


def calculate_pairwise_correlation(ticker_label_map, frequency="D", inflation_data=None):
    """
    Calculates the pairwise correlation between each pair of ticker symbols using the longest possible time available
    and resampling to the specified frequency. Uses inflation-adjusted values if SUBTRACT_INFLATION_FLAG is True.

    Args:
        ticker_label_map (dict): A dictionary mapping ticker symbols to their labels.
        frequency (str): The frequency for resampling the data ("D" for daily, "W" for weekly, "M" for monthly).
        inflation_data (dict): A dictionary mapping dates to cumulative inflation factors.

    Returns:
        pd.DataFrame: A DataFrame containing the pairwise correlation matrix with labels as row and column names.
    """
    closing_prices = {}
    date_sets = {}

    # Fetch historical data and collect dates for each ticker
    for ticker, label in ticker_label_map.items():
        history = get_asset_stats(ticker)
        if history:
            # Determine the earliest date in the historical data
            earliest_date = min(datetime.fromisoformat(record["timestamp"]).date() for record in history)

            # Precompute inflation factors
            if SUBTRACT_INFLATION_FLAG and inflation_data is not None:
                inflation_factors = precompute_inflation_factors(inflation_data, earliest_date)
                history = adjust_for_inflation(history, inflation_factors)

            # Extract dates and store them in a set
            dates = {datetime.fromisoformat(record["timestamp"]).date() for record in history}
            date_sets[label] = dates

            # Store the full history temporarily
            closing_prices[label] = {
                datetime.fromisoformat(record["timestamp"]).date(): (
                    record["data"].get("Inflation Adjusted Close", record["data"]["Close"])  # Use "Close" if "Inflation Adjusted Close" is missing
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

    # Fetch yearly inflation data
    inflation_data = fetch_yearly_inflation()

    # Calculate and print daily correlation
    print("\nDaily Pairwise Correlation Matrix:")
    daily_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="D", inflation_data=inflation_data)
    print(daily_correlation)

    # Calculate and print weekly correlation
    print("\nWeekly Pairwise Correlation Matrix:")
    weekly_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="W", inflation_data=inflation_data)
    print(weekly_correlation)

    # Calculate and print monthly correlation
    print("\nMonthly Pairwise Correlation Matrix:")
    monthly_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="M", inflation_data=inflation_data)
    print(monthly_correlation)

    # Calculate and print 6-month correlation
    print("\n6-Month Pairwise Correlation Matrix:")
    six_month_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="6M", inflation_data=inflation_data)
    print(six_month_correlation)

    # Calculate and print yearly correlation
    print("\nYearly Pairwise Correlation Matrix:")
    year_correlation = calculate_pairwise_correlation(ticker_label_map, frequency="Y", inflation_data=inflation_data)
    print(year_correlation)


if __name__ == "__main__":
    main()