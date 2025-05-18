import json
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

END_DATE = 2024  # End date for inflation data

def fetch_yearly_inflation():   #Source; https://www.usinflationcalculator.com/inflation/historical-inflation-rates/
    """
    Loads yearly inflation data from the inflation.csv file.

    Returns:
        pd.DataFrame: A DataFrame containing the yearly inflation data.
    """
    CSV_FILE = "inflation.csv"  #Source; https://www.usinflationcalculator.com/inflation/historical-inflation-rates/

    try:
        # Load data from inflation.csv
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

    # Helper function to determine the number of days in a year
    def days_in_year(year):
        return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    # Iterate over each year and compute cumulative inflation factors
    for year in range(start_date.year, min(END_DATE, inflation_data.index.max().year) + 1):
        if year in inflation_data.index.year:
            inflation_rate = inflation_data.loc[inflation_data.index[inflation_data.index.year == year], "Inflation Rate"].values[0] / 100
            days = days_in_year(year)  # Get the number of days in the current year
            daily_inflation_factor = (1 + inflation_rate) ** (1 / days)

            # Compute inflation factors for each day in the year
            for day in range(1, days + 1):
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
    if not inflation_factors:
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
                raise ValueError(f"Date {date} not found in inflation factors.")

            adjusted_data.append(record)
        except Exception as e:
            print(f"Error adjusting for inflation on {record['timestamp']}: {e}")
            continue

    return adjusted_data