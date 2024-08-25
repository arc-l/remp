import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
fig, ax = plt.subplots(dpi=500)
ax.set_facecolor('gainsboro')
plt.axis('equal')
obs = np.array([
	[-0.155417, 0.005380],
	[-0.019592, 0.055827],
	[0.056904, 0.055987],
	[0.086366, 0.066930],
	[0.109117, 0.005676],
	[0.079655, -0.005267],
	[0.021601, -0.055081],
	[-0.114224, -0.105528],
	[-0.155417, 0.005380],
])
ax.plot(obs[:, 0], obs[:, 1], '-o', color='darkcyan', markersize=1.0, linewidth=1.0)
plt.axis('equal')
plt.show()
