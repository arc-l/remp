import rvg, os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from constants import *

vertices = [
    rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 0]),
    rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 0]),
    rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 1]),
    rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 1]),
    ]

boundary = rvg.polygon(vertices, False)
# boundary.draw("tmp.png", "", True)
# os.remove("tmp.png")

vertices = [
    np.array([BOUNDARY[0, 0], BOUNDARY[1, 0]]),
    np.array([BOUNDARY[0, 1], BOUNDARY[1, 0]]),
    np.array([BOUNDARY[0, 1], BOUNDARY[1, 1]]),
    np.array([BOUNDARY[0, 0], BOUNDARY[1, 1]]),
]

boundary = Polygon(vertices)
x, y = boundary.exterior.xy
print(len(x))

plt.plot(*boundary.exterior.xy)
plt.show()