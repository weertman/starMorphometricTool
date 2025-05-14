import matplotlib

matplotlib.use("QtAgg")  # Use Qt backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PySide6.QtCore import Signal, Qt


class PolarCanvas(FigureCanvas):
    """
    An interactive Matplotlib canvas for the polar plot.
    SHIFT-click removes nearest peak; normal click adds a new peak.
    """
    peaksChanged = Signal(np.ndarray)  # Emitted whenever peaks are updated

    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        super().__init__(fig)
        self.setParent(parent)

        self.angles = np.array([])
        self.distances = np.array([])
        self.peaks = np.array([])

        # Flip the polar axis so that "up" in the image is "up" in the plot
        self.ax.set_theta_direction(-1)  # Make angles go clockwise
        self.ax.set_theta_offset(0)  # Put 0° at the top

        self.ax.set_title("Interactive Polar Plot")
        self.mpl_connect('button_press_event', self.on_click)

    def set_data(self, angles, distances, peaks):
        """
        Set the data to display in the polar plot.

        Args:
            angles: Array of angles (radians)
            distances: Array of distances
            peaks: Array of peak indices
        """
        self.angles = angles
        self.distances = distances
        self.peaks = peaks
        self.update_plot()

    def on_click(self, event):
        """
        Handle mouse clicks.
        - Normal click: add a new peak near the clicked angle
        - Shift+click: remove the nearest peak
        """
        if event.inaxes != self.ax:
            return

        angle_clicked = event.xdata

        if event.guiEvent.modifiers() & Qt.ShiftModifier:
            # SHIFT => remove nearest peak
            if self.peaks.size == 0:
                return
            peak_angles = self.angles[self.peaks.astype(int)]
            diffs = np.abs((peak_angles - angle_clicked + np.pi) % (2 * np.pi) - np.pi)
            nearest_idx = np.argmin(diffs)
            self.peaks = np.delete(self.peaks, nearest_idx)
        else:
            # normal click => add
            angle_diffs = np.abs((self.angles - angle_clicked + np.pi) % (2 * np.pi) - np.pi)
            new_idx = np.argmin(angle_diffs)
            if new_idx not in self.peaks:
                self.peaks = np.append(self.peaks, new_idx)

        self.update_plot()
        self.peaksChanged.emit(self.peaks)

    def update_plot(self):
        """Update the polar plot with current data."""
        self.ax.clear()
        self.ax.set_title("Interactive Polar Plot")

        # We must re-apply the direction + offset after clearing:
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_offset(0)

        # Draw the main distance profile as before
        if len(self.angles) > 0 and len(self.angles) == len(self.distances):
            self.ax.plot(self.angles, self.distances, label='Distance Profile')

        # --- Now draw each arm tip as a number ---
        if hasattr(self, 'arm_angles') and len(self.arm_angles) > 0:
            for angle, r, lbl, color in zip(self.arm_angles,
                                            self.arm_dists,
                                            self.arm_labels,
                                            self.arm_colors):
                self.ax.text(
                    angle,
                    r,
                    str(lbl),
                    color=color,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
                self.ax.scatter(angle, r, color=color, s=20)

        self.draw()  # redraw canvas

    def set_arm_labels(self, angles, distances, labels, colors):
        """
        Tell the polar plot exactly which angles/distances to plot,
        and which number and color each tip should have.

        Args:
            angles: Array of angles (radians)
            distances: Array of distances
            labels: List of arm numbers
            colors: List of colors for each arm
        """
        self.arm_angles = np.array(angles)
        self.arm_dists = np.array(distances)
        self.arm_labels = labels  # list of integers
        self.arm_colors = colors  # list of e.g. 'red' or 'blue'
        self.update_plot()