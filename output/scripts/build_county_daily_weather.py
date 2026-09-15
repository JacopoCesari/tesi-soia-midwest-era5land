"""Aggregate already downloaded ERA5-Land data without network access."""

from soybean_yield_forecasting.era5.cli import download_main

if __name__ == "__main__":
    download_main(aggregate_only=True)
