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
	[-0.360000, -0.760000],
	[0.420000, -0.760000],
	[0.420000, -0.240000],
	[-0.360000, -0.240000],
	[-0.360000, -0.760000],
])
ax.plot(obs[:, 0], obs[:, 1], '-o', color='darkcyan', markersize=1.0, linewidth=1.0)
plt.axis('equal')
plt.savefig('tmp.png', dpi=500, bbox_inches='tight')
plt.show()
