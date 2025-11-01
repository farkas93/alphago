import numpy as np
import matplotlib.pyplot as plt
from config import BOARD_SIZE
from game.renderer import _compute_cell_size

# Generate data for plotting
board_sizes = np.arange(1, 31)
cell_sizes = [ _compute_cell_size(bs) for bs in board_sizes]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(board_sizes, cell_sizes, marker='o', linestyle='-')
plt.title('`_compute_cell_size` vs. `BOARD_SIZE`')
plt.xlabel('BOARD_SIZE')
plt.ylabel('Computed Cell Size')
plt.grid(True)
plt.xticks(np.arange(0, 31, 2))
plt.yticks(np.arange(20, 80, 5))
plt.axhline(y=30, color='r', linestyle='--', label='Min Size (30)')
plt.axhline(y=70, color='g', linestyle='--', label='Max Size (70)')
plt.legend()
plt.show()