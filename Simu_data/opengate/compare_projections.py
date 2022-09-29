import itk
import matplotlib.pyplot as plt
import numpy as np



proj_garf_2e7 = itk.array_from_image(itk.imread('activity_scaling/garf_proj_2e7_scaled_5e8.mha'))[2,:,:]
proj_garf_4e7 = itk.array_from_image(itk.imread('activity_scaling/garf_proj_4e7_scaled_5e8.mha'))[2,:,:]
proj_mc = itk.array_from_image(itk.imread('activity_scaling/proj_MC_5e8.mha'))[2,:,:]
proj_pvfree = itk.array_from_image(itk.imread('activity_scaling/proj_analytical_temp_PVfree.mhd'))[0,:,:]


print(f'mean in mc: {np.mean([proj_mc[proj_pvfree==64]])}')
print(f'mean in garf 2e7: {np.mean([proj_garf_2e7[proj_pvfree==64]])}')
print(f'mean in garf 4e7: {np.mean([proj_garf_4e7[proj_pvfree==64]])}')
print(f'std in mc: {np.std([proj_mc[proj_pvfree==64]])}')
print(f'std in garf 2e7: {np.std([proj_garf_2e7[proj_pvfree==64]])}')
print(f'std in garf 4e7: {np.std([proj_garf_4e7[proj_pvfree==64]])}')


fig, ax = plt.subplots()
# ax.plot(proj_pvfree[64,:], color = 'black', label = 'pvfree')
ax.plot(proj_mc[60,:], color = 'green', label = 'MC 1e8')
ax.plot(proj_garf_2e7[60,:], color = 'blue', label = 'garf 2e7 to 1e8')
ax.plot(proj_garf_4e7[60,:], color = 'red', label = 'garf 4e7 to 1e8')
ax.legend()




fig2,ax2 = plt.subplots()
bins = np.linspace(20, 250)
ax2.hist(proj_mc[proj_pvfree==64], bins = bins, color = 'green',alpha = 1, label = 'MC 1e8')
# ax2.hist(proj_garf_2e7[proj_pvfree==64], bins = bins, color = 'blue',alpha = 1, label = 'garf 2e7 to 1e8')
ax2.hist(proj_garf_4e7[proj_pvfree==64], bins = bins, color = 'red',alpha = 1, label = 'garf 4e7 to 1e8')
ax2.legend()
plt.show()