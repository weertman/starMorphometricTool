import numpy as np


def convert_to_datetime64(date_string):
    """
    Convert a date string in format 'mm_dd_yyyy' to a numpy.datetime64 object.

    Args:
        date_string (str): Date string in format 'mm_dd_yyyy'

    Returns:
        numpy.datetime64: The converted date

    Raises:
        ValueError: If the string doesn't match the expected format
    """
    try:
        # Split the string by underscore
        parts = date_string.split('_')

        # Validate we have exactly 3 parts
        if len(parts) != 3:
            raise ValueError(f"Expected format 'mm_dd_yyyy', got: {date_string}")

        # Extract month, day, year
        month, day, year = parts

        # Validate parts are numeric and the expected length
        if not (month.isdigit() and day.isdigit() and year.isdigit()):
            raise ValueError("Month, day, and year must be numeric")

        if len(year) != 4:
            raise ValueError(f"Year must be 4 digits, got: {year}")

        # Convert to standard ISO format (yyyy-mm-dd)
        iso_format = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Convert to numpy.datetime64
        return np.datetime64(iso_format)

    except Exception as e:
        raise ValueError(f"Failed to convert '{date_string}' to datetime64: {str(e)}")