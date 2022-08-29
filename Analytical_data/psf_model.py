import numpy as np
import matplotlib.pyplot as plt




# Units in mm

d = 1.5
l = 35
mu = 1.91*11.35*10
leff = l - 2/mu
b = 380

FWHM = d*(leff + b)/leff
beta = FWHM/(2*np.sqrt(np.log(2)))
print(f'FWHM = {FWHM} mm')


# Detector size : dX x dY (mm x mm)
dX = 546
dY = 406

sigma0 = d / (2*np.sqrt(2*np.log(2)))
alpha = sigma0 / leff

print(f'sigma0 : {sigma0}')
print(f'alpha : {alpha}')

lX = np.arange(-dX/2, dX/2, 4)
lY = np.arange(-dY/2, dY/2, 4)
X, Y = np.meshgrid(lX, lY)

R = np.sqrt(X**2 + Y**2)
PSF = np.exp(-R**2/beta**2)

sigma0 = d/(2*np.sqrt(np.log(2)))
print(f'sigma_O = {sigma0}')
alpha = sigma0/leff
print(f'alpha = {alpha}')

# 3D Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, PSF)
plt.show()


# # 2D plot
# fig, ax = plt.subplots()
# ax.plot(lX, np.exp(-lX**2/beta**2))
# plt.show()