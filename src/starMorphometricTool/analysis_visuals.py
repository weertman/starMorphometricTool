"""
analysis_visuals.py

An updated collection of functions to visualize and summarize star morphometrics data
with explicit date parsing and safer handling of small datasets or mismatched lengths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress


def convert_to_datetime(df, date_col='measurement_date'):
    """
    Convert a string-based 'measurement_date' column (e.g. '01_24_2025')
    into a proper datetime. We assume the format is '%m_%d_%Y'.

    Any rows that fail to parse become 'NaT'.
    """
    if date_col not in df.columns:
        return df  # nothing to do

    try:
        # Force the specific format mm_dd_yyyy
        df[date_col] = pd.to_datetime(
            df[date_col],
            format='%m_%d_%Y',
            errors='coerce'  # rows that don't match get NaT
        )
    except Exception as e:
        print(f"[WARNING] Could not parse '{date_col}' with '%m_%d_%Y': {e}")
    return df


def linear_regression_against_time(dates, values):
    """
    Given two arrays/Series: 'dates' (datetime) and 'values' (float),
    perform a linear regression of 'values' on "days since earliest date".

    Returns:
        slope, intercept, r_squared, p_value
        (or np.nan for all if there's insufficient data).
    """
    # Combine into a single DataFrame to align them properly
    tmp_df = pd.DataFrame({'dates': dates, 'values': values})
    # Drop rows where either is NaN or NaT
    tmp_df.dropna(subset=['dates', 'values'], inplace=True)

    # If fewer than 2 points remain, return NaNs
    if len(tmp_df) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Convert 'dates' to real datetime if not already
    tmp_df['dates'] = pd.to_datetime(tmp_df['dates'], errors='coerce')
    tmp_df.dropna(subset=['dates'], inplace=True)
    if len(tmp_df) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Calculate days since earliest date
    min_date = tmp_df['dates'].min()        # a Timestamp
    # Convert each date to (days from min_date), as float
    tmp_df['days_from_start'] = (tmp_df['dates'] - min_date) / pd.Timedelta(days=1)

    # Need at least 2 valid numeric points
    if tmp_df['days_from_start'].nunique() < 1:
        return np.nan, np.nan, np.nan, np.nan

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        tmp_df['days_from_start'].values,
        tmp_df['values'].values
    )

    return slope, intercept, r_value**2, p_value


def rainshadow_plot(ax,
                    data,
                    x_position=0,
                    width=0.4,
                    facecolor='skyblue',
                    edgecolor='black',
                    alpha=0.6,
                    scatter_color='black',
                    scatter_alpha=0.5,
                    scatter_size=10,
                    boxplot_offset=-0.2):
    """
    Creates a 'rainshadow' plot (half-violin + boxplot + jittered scatter) at x_position.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return  # no valid data

    # Compute kernel density estimate
    kde = gaussian_kde(data)
    y_vals = np.linspace(data.min(), data.max(), 200)
    pdf = kde(y_vals)

    # Scale to desired max width
    pdf = pdf / pdf.max() * width

    # Plot the half-violin
    ax.fill_betweenx(y_vals,
                     x_position,
                     x_position + pdf,
                     facecolor=facecolor,
                     edgecolor=edgecolor,
                     alpha=alpha)

    # Boxplot offset to the left
    bp = ax.boxplot(
        data,
        positions=[x_position + boxplot_offset],
        widths=0.2,
        patch_artist=True
    )
    for patch in bp['boxes']:
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_alpha(alpha)

    # Add jittered scatter
    jitter = np.random.rand(len(data)) * (width * 0.3)
    ax.scatter(
        x_position + jitter + 0.05,
        data,
        color=scatter_color,
        alpha=scatter_alpha,
        s=scatter_size,
        edgecolors='none'
    )


def plot_rainshadow_distributions(df,
                                  columns=None,
                                  date_col='measurement_date'):
    """
    Creates a set of subplots with 'rainshadow' distributions for each
    measurement date in `df`, for each column in `columns`.

    By default, tries columns:
      'mean_width', 'median_width', 'std_width', 'area_mm2',
      'major_axis_length_mm', 'minor_axis_length_mm'
    """
    if columns is None:
        columns = [
            'mean_width',
            'median_width',
            'std_width',
            'area_mm2',
            'major_axis_length_mm',
            'minor_axis_length_mm'
        ]

    # Parse the date_col with known format
    df = convert_to_datetime(df, date_col=date_col)

    # Identify unique measurement dates (sorted by date)
    # Drop rows with NaT in date_col
    df_valid = df.dropna(subset=[date_col])
    if df_valid.empty:
        # If no valid dates, just do an empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid dates to plot", ha='center', va='center')
        return fig, [ax]

    unique_dates = sorted(df_valid[date_col].unique())

    # We'll make subplots in a grid
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.5*n_rows), squeeze=False)
    axs = axs.ravel()

    for idx, col in enumerate(columns):
        ax = axs[idx]
        ax.set_title(col)

        for i, d_val in enumerate(unique_dates):
            subset = df_valid[df_valid[date_col] == d_val]
            data = subset[col].values if col in subset.columns else np.array([])

            rainshadow_plot(ax, data, x_position=i, facecolor='blue', edgecolor='black')

        # x-axis labeling
        ax.set_xticks(range(len(unique_dates)))
        # Show date strings in YYYY-MM-DD or whichever you prefer
        labels = [str(pd.to_datetime(d).date()) for d in unique_dates]
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Remove any unused subplots
    for k in range(len(columns), len(axs)):
        fig.delaxes(axs[k])

    fig.tight_layout()
    return fig, axs


def plot_regression_lines(df,
                          columns=None,
                          date_col='measurement_date'):
    """
    For each variable in `columns`, run a linear regression vs. time
    (days from earliest date). Print the results or skip if insufficient data.

    Updated: we skip if there's only one unique date (no time variation).
    """
    if columns is None:
        columns = [
            'mean_width',
            'median_width',
            'std_width',
            'area_mm2',
            'major_axis_length_mm',
            'minor_axis_length_mm'
        ]

    # Ensure we parse the date column
    df = convert_to_datetime(df, date_col=date_col)

    print("\n=== Linear Regression vs. Days from Earliest Date ===\n")

    for col in columns:
        if col not in df.columns:
            print(f"[SKIP] Column '{col}' not in DataFrame.")
            continue

        # Build a sub-DataFrame for alignment
        sub_df = df[[date_col, col]].dropna()
        if sub_df.empty:
            print(f"[SKIP] Column '{col}' has no valid data.")
            continue

        # Check how many unique dates
        unique_dates = sub_df[date_col].unique()
        if len(unique_dates) < 2:
            # If there's only one date, we skip regression
            print(f"  [SKIP] Column '{col}': only one unique date, cannot fit time-based regression.")
            continue

        # Perform the regression
        slope, intercept, r2, pval = linear_regression_against_time(
            sub_df[date_col], sub_df[col]
        )

        # If slope is nan => not enough data or something else
        if np.isnan(slope):
            print(f"[SKIP] Column '{col}' does not have >=2 valid points for regression.")
            continue

        print(f"  {col}: y = {slope:.4f} * days + {intercept:.4f}")
        print(f"    R²   = {r2:.4f}")
        print(f"    p-val= {pval:.4e}\n")

