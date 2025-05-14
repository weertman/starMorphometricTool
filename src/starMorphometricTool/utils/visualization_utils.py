import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA


def visualize_pca_with_images(normalized_images, max_thumbnails=None, max_image_size=0.05,
                              fig_size=(12, 10), alpha=0.9, pca_components=2,
                              marker_size=20, min_dist_factor=0.1, bg_threshold=5,
                              enforce_max_thumbnails=True, title="PCA of Normalized Objects with Image Overlays"):
    """
    Create a PCA visualization with actual image thumbnails at their PCA coordinates.

    Args:
        normalized_images (list): List of normalized images (all same size)
        max_thumbnails (int): Maximum number of thumbnails to display (None=all)
        max_image_size (float): Maximum fraction of plot size for thumbnails (0.0-1.0)
        fig_size (tuple): Figure size (width, height)
        alpha (float): Alpha transparency for the thumbnails
        pca_components (int): Number of PCA components to compute
        marker_size (int): Size of scatter markers for points without thumbnails
        min_dist_factor (float): Minimum distance factor between thumbnails (smaller=more images)
        bg_threshold (int): Threshold for determining background pixels (0-255)
        enforce_max_thumbnails (bool): If True, ensures at least max_thumbnails are shown
        title (str): Title for the plot

    Returns:
        fig, ax: The figure and axis with the visualization
    """
    if not normalized_images:
        raise ValueError("No images provided for visualization")

    # Prepare data for PCA
    X = []
    for img in normalized_images:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # RGB
                gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img[:, :, 0]  # Use first channel
        else:
            gray = img
        # Flatten the image
        X.append(gray.flatten())

    # Create a matrix where each row is a flattened image
    X = np.array(X)

    # Perform PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # First plot all points as a scatter plot
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], s=marker_size, alpha=0.5)

    # Determine how many thumbnails to show
    n_images = len(normalized_images)
    if max_thumbnails is None or max_thumbnails > n_images:
        max_thumbnails = n_images

    # For large datasets, use a smarter selection to cover the PCA space better
    if max_thumbnails < n_images:
        # Use a stratified selection to ensure coverage across the PCA space
        # First, calculate ranges for binning
        x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
        y_range = X_pca[:, 1].max() - X_pca[:, 1].min()

        # Determine number of bins in each dimension
        n_bins = max(2, int(np.sqrt(max_thumbnails)))

        # Create bins
        x_bins = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), n_bins + 1)
        y_bins = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), n_bins + 1)

        # Assign each point to a bin
        x_bin_indices = np.digitize(X_pca[:, 0], x_bins) - 1
        y_bin_indices = np.digitize(X_pca[:, 1], y_bins) - 1

        # Ensure they're within valid range
        x_bin_indices = np.clip(x_bin_indices, 0, n_bins - 1)
        y_bin_indices = np.clip(y_bin_indices, 0, n_bins - 1)

        # Create 2D bin assignments
        bin_indices = y_bin_indices * n_bins + x_bin_indices

        # Count points per bin
        unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)

        # Calculate proportions
        bin_props = bin_counts / bin_counts.sum()

        # Calculate how many samples to take from each bin
        samples_per_bin = np.round(bin_props * max_thumbnails).astype(int)

        # Ensure we don't exceed the counts
        for i, (bin_idx, count) in enumerate(zip(unique_bins, bin_counts)):
            if samples_per_bin[i] > count:
                samples_per_bin[i] = count

        # Adjust to hit max_thumbnails exactly
        total_samples = samples_per_bin.sum()
        if total_samples < max_thumbnails:
            # Add samples to most populous bins
            sorted_bins = np.argsort(bin_counts)[::-1]
            for i in sorted_bins:
                bin_idx = unique_bins[i]
                if samples_per_bin[i] < bin_counts[i]:
                    samples_per_bin[i] += 1
                    total_samples += 1
                    if total_samples >= max_thumbnails:
                        break

        # Select samples from each bin
        selected_indices = []
        for bin_idx, n_samples in zip(unique_bins, samples_per_bin):
            if n_samples <= 0:
                continue

            # Find points in this bin
            bin_points = np.where(bin_indices == bin_idx)[0]

            # Randomly select n_samples from this bin
            if len(bin_points) > n_samples:
                selected_from_bin = np.random.choice(bin_points, int(n_samples), replace=False)
            else:
                selected_from_bin = bin_points

            selected_indices.extend(selected_from_bin)

        # Ensure we have enough samples
        if len(selected_indices) < max_thumbnails and enforce_max_thumbnails:
            # Add more random samples if needed
            available = list(set(range(n_images)) - set(selected_indices))
            if available:
                n_additional = min(max_thumbnails - len(selected_indices), len(available))
                additional = np.random.choice(available, n_additional, replace=False)
                selected_indices.extend(additional)

        indices = np.array(selected_indices)
    else:
        indices = np.arange(n_images)

    # Calculate minimum distance threshold - much smaller than before
    x_range = X_pca[:, 0].max() - X_pca[:, 0].min()
    y_range = X_pca[:, 1].max() - X_pca[:, 1].min()
    min_dist_threshold = min_dist_factor * min(x_range, y_range) / np.sqrt(max_thumbnails)

    # Create a KD-tree for nearest neighbor search
    from scipy.spatial import KDTree
    if len(X_pca) > 1:
        tree = KDTree(X_pca)

    # Calculate appropriate thumbnail size based on data range and max_image_size
    # This defines how large the image appears relative to the plot area
    zoom_factor = max_image_size * min(fig_size) / max(1, np.sqrt(max_thumbnails))

    # Create thumbnails with transparent backgrounds
    added_indices = []
    for i in indices:
        # Skip if too close to already added thumbnails
        if len(added_indices) > 0:
            # Find distance to closest already-added point
            distances = np.sqrt(np.sum((X_pca[i] - X_pca[added_indices]) ** 2, axis=1))
            min_dist = np.min(distances)
            if min_dist < min_dist_threshold:
                continue

        # Add current index to added_indices
        added_indices.append(i)

        # Create RGBA version of the image with transparent background
        img = normalized_images[i]
        if len(img.shape) == 3 and img.shape[2] == 3:  # RGB
            # Create transparency mask by checking if pixel is close to black
            is_bg = np.all(img < bg_threshold, axis=2)

            # Convert to RGBA
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = img

            # Set alpha channel to 0 for background (black) pixels
            rgba[:, :, 3] = np.where(is_bg, 0, 255 * alpha)
        else:
            # For grayscale, convert to RGBA
            is_bg = img < bg_threshold

            # Create RGBA
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = np.dstack([img] * 3)
            rgba[:, :, 3] = np.where(is_bg, 0, 255 * alpha)

        # Add the image to the plot
        imagebox = offsetbox.OffsetImage(rgba, zoom=zoom_factor)
        ab = offsetbox.AnnotationBbox(
            imagebox,
            (X_pca[i, 0], X_pca[i, 1]),
            frameon=False,
            pad=0
        )
        ax.add_artist(ab)

    # Add labels and information
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Make axes equal to preserve aspect ratio
    ax.set_aspect('equal')

    # Add a legend
    if max_thumbnails < n_images:
        plt.figtext(0.02, 0.02, f'Showing {len(added_indices)} of {n_images} images',
                    ha='left', va='bottom', fontsize=10)

    return fig, ax, X_pca, pca