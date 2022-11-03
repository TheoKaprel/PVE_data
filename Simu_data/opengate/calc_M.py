import itk
import matplotlib.pyplot as plt
import numpy as np



# proj_garf1 = itk.array_from_image(itk.imread('activity_scaling/garf_1e8.mha'))[2, :, :]
# proj_garf2 = itk.array_from_image(itk.imread('activity_scaling/garf_1e8_batch1e5.mha'))[2, :, :]
#
#
# mean_proj = (np.mean(proj_garf1)+np.mean(proj_garf2))/2
#
# r_eff = 0.99
# var_i = np.var(proj_garf1)
#
# M = r_eff * var_i / (mean_proj * (1 - r_eff))
#
#
# print(M)



proj_mc = itk.array_from_image(itk.imread('activity_scaling/proj_5e8.mhd'))[2,:,:]
proj_pvfree = itk.array_from_image(itk.imread('activity_scaling/big_square_proj_analytical_PVfree.mha'))[0,:,:]

proj_garf_5e7= itk.array_from_image(itk.imread('activity_scaling/garf_5e7_scaled_to_5e8.mha'))[2,:,:]
proj_garf_1e8 = itk.array_from_image(itk.imread('activity_scaling/garf_1e8_scaled_to_5e8.mha'))[2, :, :]
proj_garf_1_3e8 = itk.array_from_image(itk.imread('activity_scaling/garf_1.3e8_scaled_to_5e8.mha'))[2, :, :]
proj_garf_2e8 = itk.array_from_image(itk.imread('activity_scaling/garf_2e8_scaled_to_5e8.mha'))[2, :, :]
proj_garf_5e8 = itk.array_from_image(itk.imread('activity_scaling/garf_5e8.mha'))[2,:,:]

inside = proj_pvfree==296
var_mc = np.var(proj_mc[inside])

list_proj = [proj_garf_5e7, proj_garf_1e8,proj_garf_1_3e8, proj_garf_2e8, proj_garf_5e8]
list_var = [np.var(proj[inside]) for proj in list_proj]

print(list_var)

fig,ax = plt.subplots()
ax.plot([5e7,1e8,1.3e8,2e8,5e8], list_var,'-o')
ax.axhline(var_mc, color = 'grey', linestyle = 'dashed', label="Variance MC")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('# primaries')
ax.set_ylabel('Variance garf')
plt.legend()
plt.show()