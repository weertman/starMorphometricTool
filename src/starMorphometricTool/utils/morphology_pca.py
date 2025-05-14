import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_morphology_pca(df, columns=None, n_components=2):
    """
    Compute Principal Component Analysis on the specified numerical columns.

    Args:
        df (pandas.DataFrame): DataFrame containing morphometric measurements
        columns (list, optional): List of column names to include in the PCA.
                                 If None, uses all numeric columns except id and date
        n_components (int, optional): Number of principal components to compute

    Returns:
        tuple: (pca_result, pca_model, scaler, feature_names)
            - pca_result: DataFrame with id, measurement_day, and the principal components
            - pca_model: Fitted sklearn PCA model
            - scaler: Fitted sklearn StandardScaler
            - feature_names: List of features used in the PCA
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # If columns not specified, use all numeric columns except id and date
    if columns is None:
        exclude_cols = ['id', 'measurement_day', 'morphometrics_path']
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in exclude_cols]

    # Check that all requested columns exist
    missing_cols = [col for col in columns if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in the DataFrame: {missing_cols}")

    # Extract the features for PCA
    X = df_copy[columns].values

    # Standardize the features (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create column names for the principal components
    pc_names = [f'PC{i + 1}' for i in range(n_components)]

    # Create a DataFrame with the results
    pca_df = pd.DataFrame(data=principal_components, columns=pc_names)

    # Add identification columns
    pca_df['id'] = df_copy['id'].values
    if 'measurement_day' in df_copy.columns:
        pca_df['measurement_day'] = df_copy['measurement_day'].values

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100

    # Print explained variance
    print(f"Explained variance by principal components:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i + 1}: {var:.2f}%")

    # Print component loadings
    print("\nComponent loadings:")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(n_components)],
        index=columns
    )
    print(loadings)

    return pca_df, pca, scaler, columns


def visualize_morphology_pca(pca_df, color_by=None, title="PCA of Morphometric Measurements"):
    """
    Create a scatter plot of the first two principal components.

    Args:
        pca_df (pandas.DataFrame): DataFrame containing PC1 and PC2 (from compute_morphology_pca)
        color_by (str, optional): Column name to color points by (e.g., 'id' or 'measurement_day')
        title (str, optional): Title for the plot

    Returns:
        tuple: (fig, ax) - The figure and axis objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Check if color_by is specified and exists in the DataFrame
    if color_by is not None and color_by in pca_df.columns:
        scatter = sns.scatterplot(
            x='PC1', y='PC2',
            hue=color_by,
            data=pca_df,
            s=100,
            alpha=0.8,
            ax=ax
        )

        # If there are many unique values, adjust the legend
        if pca_df[color_by].nunique() > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        scatter = sns.scatterplot(
            x='PC1', y='PC2',
            data=pca_df,
            s=100,
            alpha=0.8,
            ax=ax
        )

    # Add labels
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add annotations for each point (optional)
    for i, row in pca_df.iterrows():
        ax.text(row['PC1'], row['PC2'], row['id'], fontsize=8)

    return fig, ax


def plot_pca_by_individual(pca_df, min_measurements=2):
    """
    Create a plot showing how individuals changed between measurements.

    Args:
        pca_df (pandas.DataFrame): DataFrame from compute_morphology_pca
        min_measurements (int): Minimum number of measurements per individual to include

    Returns:
        tuple: (fig, ax) - The figure and axis objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(12, 8))

    # Count measurements per individual
    counts = pca_df['id'].value_counts()
    valid_ids = counts[counts >= min_measurements].index

    # Filter to individuals with multiple measurements
    multi_df = pca_df[pca_df['id'].isin(valid_ids)].copy()

    # Sort by measurement day to ensure correct line ordering
    multi_df = multi_df.sort_values(['id', 'measurement_day'])

    # Plot each individual
    for individual in valid_ids:
        # Get data for this individual
        ind_data = multi_df[multi_df['id'] == individual]

        # Plot points
        ax.scatter(
            ind_data['PC1'], ind_data['PC2'],
            label=individual, s=80, alpha=0.7
        )

        # Plot lines connecting measurements
        ax.plot(
            ind_data['PC1'], ind_data['PC2'],
            '-o', linewidth=1.5, alpha=0.5
        )

        # Add label at the last point
        last_point = ind_data.iloc[-1]
        ax.annotate(
            individual,
            (last_point['PC1'], last_point['PC2']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8
        )

    # Add labels and title
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Individual Changes Across Measurements')
    ax.grid(True, alpha=0.3)

    return fig, ax


def create_interactive_pca_plot(pca_df, color_by=None):
    """
    Create an interactive PCA plot using Plotly.

    Args:
        pca_df (pandas.DataFrame): DataFrame with PC1, PC2 from compute_morphology_pca
        color_by (str, optional): Column to use for coloring points

    Returns:
        plotly.graph_objects.Figure: Interactive figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Basic scatter plot
        if color_by and color_by in pca_df.columns:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color=color_by,
                hover_name='id',
                hover_data=['measurement_day'],
                title='Interactive PCA of Morphometric Measurements'
            )
        else:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                hover_name='id',
                hover_data=['measurement_day'],
                title='Interactive PCA of Morphometric Measurements'
            )

        # Add lines connecting same individuals on different dates
        for individual in pca_df['id'].unique():
            ind_data = pca_df[pca_df['id'] == individual].sort_values('measurement_day')
            if len(ind_data) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=ind_data['PC1'],
                        y=ind_data['PC2'],
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        showlegend=False,
                        opacity=0.5,
                        hoverinfo='skip'
                    )
                )

        # Improve layout
        fig.update_layout(
            xaxis_title='PC1',
            yaxis_title='PC2',
            legend_title=color_by if color_by else '',
            height=600,
            width=900
        )

        return fig
    except ImportError:
        print("Plotly not installed. Run 'pip install plotly' to use this function.")
        return None


def create_pca_biplot(pca_df, pca_model, feature_names, color_by=None, connect_by=None,
                      title="PCA Biplot of Morphometric Measurements",
                      scaling=1, arrow_scale=1, arrow_color='navy',
                      marker_size=80, use_viridis_for_dates=True,
                      label_offset=0.05, line_color='black', line_width=0.5):
    """
    Create a biplot showing both the PCA scores and the feature loading vectors.

    Args:
        pca_df (pandas.DataFrame): DataFrame containing PC1 and PC2 from compute_morphology_pca
        pca_model: Fitted PCA model from sklearn
        feature_names (list): Names of the features used in PCA
        color_by (str, optional): Column name to color points by
        connect_by (str, optional): Column name to connect points by (e.g., 'id')
        title (str): Title for the plot
        scaling (float): Scaling factor for the arrows (usually between 0 and 1)
        arrow_scale (float): Scale factor for arrow size
        arrow_color (str): Color for the loading vectors
        marker_size (int): Size of scatter point markers
        use_viridis_for_dates (bool): Use viridis colormap for date columns
        label_offset (float): Offset for feature labels to prevent overlap
        line_color (str): Color for lines connecting points with same ID
        line_width (float): Width for connecting lines

    Returns:
        tuple: (fig, ax) - The matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the scatter points
    if color_by is not None and color_by in pca_df.columns:
        # Special handling for dates if requested
        if use_viridis_for_dates and ('date' in color_by.lower() or 'day' in color_by.lower()):
            # Try to convert to datetime
            try:
                dates = pd.to_datetime(pca_df[color_by])
                # Create a colormap
                cmap = cm.get_cmap('viridis')
                # Normalize dates to 0-1 range
                min_date = dates.min()
                max_date = dates.max()
                date_range = (max_date - min_date).total_seconds()

                if date_range == 0:  # Handle case with only one date
                    norm_dates = np.zeros(len(dates))
                else:
                    norm_dates = [(d - min_date).total_seconds() / date_range for d in dates]

                # Plot with viridis colormap
                scatter = ax.scatter(
                    pca_df['PC1'], pca_df['PC2'],
                    c=norm_dates,
                    cmap='viridis',
                    s=marker_size,
                    alpha=0.7
                )

                # Add a colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_by)

                # Add date ticks to colorbar
                if date_range > 0:
                    num_ticks = min(5, len(dates.unique()))
                    tick_positions = np.linspace(0, 1, num_ticks)
                    tick_dates = [min_date + pd.Timedelta(seconds=pos * date_range) for pos in tick_positions]
                    tick_labels = [d.strftime('%Y-%m-%d') for d in tick_dates]
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels(tick_labels)

            except (ValueError, pd.errors.ParserError):
                # If conversion fails, treat as regular categories
                use_viridis_for_dates = False

        # Standard categorical plotting if not using viridis for dates
        if not use_viridis_for_dates or ('date' not in color_by.lower() and 'day' not in color_by.lower()):
            categories = pca_df[color_by].unique()

            # Just color by category
            for i, category in enumerate(categories):
                subset = pca_df[pca_df[color_by] == category]
                ax.scatter(
                    subset['PC1'], subset['PC2'],
                    label=category,
                    alpha=0.7,
                    s=marker_size
                )

            # Add a legend
            if len(categories) <= 20:  # Only show legend if not too many categories
                ax.legend(title=color_by, loc='best')
    else:
        # Simple scatter plot without coloring
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, s=marker_size)

    # Connect points with the same ID using fine black lines
    if connect_by is not None and connect_by in pca_df.columns:
        # Sort by measurement_day if available
        sort_col = 'measurement_day' if 'measurement_day' in pca_df.columns else None

        for id_val in pca_df[connect_by].unique():
            # Get points for this ID
            id_points = pca_df[pca_df[connect_by] == id_val].copy()

            # Sort by date/time if possible
            if sort_col:
                id_points = id_points.sort_values(sort_col)

            # Draw line connecting points
            if len(id_points) > 1:
                ax.plot(
                    id_points['PC1'],
                    id_points['PC2'],
                    '-',
                    color=line_color,
                    linewidth=line_width,
                    alpha=0.7
                )

                # Add ID label to the last point
                last_point = id_points.iloc[-1]
                ax.text(
                    last_point['PC1'],
                    last_point['PC2'],
                    str(id_val),
                    fontsize=8,
                    ha='left',
                    va='bottom'
                )

    # Get the PCA loadings (coefficients)
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    # Find overlapping vectors
    from scipy.spatial.distance import pdist, squareform

    # Calculate endpoint positions for each vector
    arrow_endpoints = np.zeros((len(feature_names), 2))
    for i in range(len(feature_names)):
        arrow_endpoints[i, 0] = loadings[i, 0] * scaling * arrow_scale
        arrow_endpoints[i, 1] = loadings[i, 1] * scaling * arrow_scale

    # Calculate distances between endpoints
    if len(feature_names) > 1:
        endpoint_distances = squareform(pdist(arrow_endpoints))

        # Create a dictionary to store offset directions for each feature
        offsets = {}

        # Find close pairs and assign opposite offsets
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                # If endpoints are close
                if endpoint_distances[i, j] < 0.2 * np.sqrt(np.sum(arrow_endpoints ** 2) / len(arrow_endpoints)):
                    # Calculate normalized perpendicular vector
                    vec = arrow_endpoints[i] - arrow_endpoints[j]
                    if np.all(vec == 0):
                        perp = np.array([label_offset, label_offset])
                    else:
                        perp = np.array([-vec[1], vec[0]])
                        perp = perp / np.linalg.norm(perp) * label_offset

                    # Assign opposite offsets
                    offsets[i] = perp
                    offsets[j] = -perp

    # Use different colors for better visibility
    import matplotlib.colors as mcolors
    arrow_colors = list(mcolors.TABLEAU_COLORS.values())

    # If single arrow color is specified, use it for all arrows
    if arrow_color != 'navy':
        arrow_colors = [arrow_color] * len(feature_names)

    # Plot the loading vectors (arrows)
    for i, feature in enumerate(feature_names):
        # Scale the loadings
        x = loadings[i, 0] * scaling * arrow_scale
        y = loadings[i, 1] * scaling * arrow_scale

        # Use the appropriate color
        color = arrow_colors[i % len(arrow_colors)]

        # Draw the arrow
        ax.arrow(0, 0, x, y, color=color, alpha=0.8, width=0.01, head_width=0.05)

        # Label the arrow
        # Position the label a bit further out from the arrow tip
        label_x = x * 1.1
        label_y = y * 1.1

        # Apply offset if needed to avoid overlap
        if i in offsets:
            label_x += offsets[i][0]
            label_y += offsets[i][1]

        ha = 'center'
        if label_x > 0:
            ha = 'left'
        elif label_x < 0:
            ha = 'right'

        va = 'center'
        if label_y > 0:
            va = 'bottom'
        elif label_y < 0:
            va = 'top'

        ax.text(label_x, label_y, feature, ha=ha, va=va, color=color, fontsize=10)

    # Add a title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%} variance)")

    # Add grid lines
    ax.axhline(y=0, color='lightgray', linestyle='--', alpha=0.8)
    ax.axvline(x=0, color='lightgray', linestyle='--', alpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Make the plot square
    ax.set_aspect('equal')

    # Set limits to make vectors visible
    max_val = max(
        abs(pca_df['PC1'].max()), abs(pca_df['PC1'].min()),
        abs(pca_df['PC2'].max()), abs(pca_df['PC2'].min())
    )
    buffer = max_val * 0.1
    ax.set_xlim(-max_val - buffer, max_val + buffer)
    ax.set_ylim(-max_val - buffer, max_val + buffer)

    return fig, ax