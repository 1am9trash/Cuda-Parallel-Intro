import re
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

l = 64
file_path = f"gpu_grid_{l}_bs8x8x8.data"

diagonal_data = []

with open(file_path, "r") as f:
    for line in f:
        match = re.match(r"\((\d+), (\d+), (\d+)\) = ([\d\.Ee+-]+)", line.strip())
        if match:
            x, y, z, val = map(float, match.groups())
            if x > l / 2:
                continue
            if x == y == z:
                r = math.sqrt((x - (l / 2)) ** 2 * 3)
                diagonal_data.append((r, val))

diagonal_data.sort()
df = pd.DataFrame(diagonal_data, columns=["Distance", "Phi Value"])
print(df)

