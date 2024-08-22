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
obs = np.array([
	[-0.324178, -0.388224],
	[-0.193215, -0.345297],
	[-0.116771, -0.348111],
	[-0.091589, -0.339857],
	[-0.074974, -0.390546],
	[-0.100156, -0.398800],
	[-0.160102, -0.446320],
	[-0.291065, -0.489247],
	[-0.324178, -0.388224],
])
ax.plot(obs[:, 0], obs[:, 1], '-o', color='darkcyan', markersize=1.0, linewidth=1.0)
obs = np.array([
	[-0.063876, -0.662822],
	[-0.063876, -0.466172],
	[0.126841, -0.424516],
	[0.123834, -0.662822],
	[-0.063876, -0.662822],
])
ax.plot(obs[:, 0], obs[:, 1], '-o', color='darkcyan', markersize=1.0, linewidth=1.0)
obs = np.array([
	[0.155608, -0.304336],
	[0.120618, -0.252631],
	[0.122243, -0.251144],
	[0.163849, -0.285051],
	[0.237327, -0.374184],
	[0.289284, -0.450961],
	[0.322292, -0.519183],
	[0.323744, -0.552791],
	[0.293085, -0.538949],
	[0.242025, -0.482943],
	[0.190068, -0.406166],
	[0.157059, -0.337944],
	[0.155608, -0.304336],
])
ax.plot(obs[:, 0], obs[:, 1], '-o', color='darkcyan', markersize=1.0, linewidth=1.0)
plt.axis('equal')
plt.show()
