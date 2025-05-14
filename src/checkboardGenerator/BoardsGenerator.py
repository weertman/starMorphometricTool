import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List

@dataclass
class CheckerboardConfig:
    rows: int
    cols: int
    square_size_mm: float
    dpi: int = 300
    paper_size: tuple = (8.5, 11)  # inches
    save_directory: str = 'checkerboards'

class CheckerboardGenerator:
    def __init__(self, configs: List[CheckerboardConfig]):
        self.configs = configs
        self.ensure_save_directory()

    def ensure_save_directory(self):
        if not os.path.exists(self.configs[0].save_directory):
            os.makedirs(self.configs[0].save_directory)
            print(f"Created directory: {self.configs[0].save_directory}")

    def create_checkerboard(self, config: CheckerboardConfig) -> np.ndarray:
        """
        Creates a checkerboard pattern based on the provided configuration.
        """
        checkerboard = 255 * ((np.arange(config.rows)[:, None] + np.arange(config.cols)) % 2 == 0).astype(np.uint8)
        return checkerboard

    def plot_and_save(self, checkerboard: np.ndarray, config: CheckerboardConfig):
        """
        Plots the checkerboard and saves it as an image with a descriptive filename.
        Ensures that each checker is square.
        """
        # Convert square size from mm to inches
        square_size_inch = config.square_size_mm / 25.4  # 1 inch = 25.4 mm

        # Calculate the size of the checkerboard in inches
        board_width_inch = config.cols * square_size_inch
        board_height_inch = config.rows * square_size_inch

        paper_width, paper_height = config.paper_size

        # Check if the checkerboard fits on the paper
        if board_width_inch > paper_width or board_height_inch > paper_height:
            print(f"Checkerboard {config.rows}x{config.cols} with square size {config.square_size_mm}mm "
                  f"does not fit on {paper_width}x{paper_height} inch paper. Skipping.")
            return

        # Calculate margins to center the checkerboard
        margin_left_right = (paper_width - board_width_inch) / 2
        margin_bottom_top = (paper_height - board_height_inch) / 2

        fig, ax = plt.subplots(figsize=config.paper_size, dpi=config.dpi)

        # Display the checkerboard with aspect='equal' to ensure squares are square
        ax.imshow(checkerboard, cmap='gray', extent=(
            margin_left_right,
            margin_left_right + board_width_inch,
            margin_bottom_top,
            margin_bottom_top + board_height_inch
        ), aspect='equal')

        ax.axis('off')  # Hide axes

        # Create descriptive filename
        filename = f"calibration_board_{config.rows}x{config.cols}_{config.square_size_mm}mm.png"
        save_path = os.path.join(config.save_directory, filename)

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=config.dpi)
        plt.close(fig)
        print(f"Saved: {save_path}")

    def generate_all(self):
        """
        Generates and saves all checkerboards based on the provided configurations.
        """
        for config in self.configs:
            checkerboard = self.create_checkerboard(config)
            self.plot_and_save(checkerboard, config)

def get_predefined_configs() -> List[CheckerboardConfig]:
    """
    Returns a list of predefined checkerboard configurations.
    """
    configs = [
        CheckerboardConfig(rows=5, cols=7, square_size_mm=25),
        CheckerboardConfig(rows=15, cols=16, square_size_mm=5),
    ]
    return configs

if __name__ == '__main__':
    # Retrieve predefined configurations
    configurations = get_predefined_configs()

    # Initialize the generator with configurations
    generator = CheckerboardGenerator(configurations)

    # Generate and save all checkerboards
    generator.generate_all()
