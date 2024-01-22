import itk
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from scipy.special import factorial


proj_garf_2e8 = itk.array_from_image(itk.imread('activity_scaling/garf_2e8_scaled_to_5e8.mha'))[2, :, :]
proj_mc = itk.array_from_image(itk.imread('activity_scaling/proj_5e8.mhd'))[2,:,:]
proj_pvfree = itk.array_from_image(itk.imread('activity_scaling/big_square_proj_analytical_PVfree.mha'))[0,:,:]
proj_garf_5e8 = itk.array_from_image(itk.imread('activity_scaling/garf_5e8.mha'))[2,:,:]
proj_garf_5e7= itk.array_from_image(itk.imread('activity_scaling/garf_5e7_scaled_to_5e8.mha'))[2,:,:]
proj_garf_1_3e8= itk.array_from_image(itk.imread('activity_scaling/garf_1.3e8_scaled_to_5e8.mha'))[2,:,:]

proj_garf_5e7_poisson = np.random.poisson(lam=proj_garf_5e7, size = proj_garf_5e7.shape)
proj_garf_2e8_poisson = np.random.poisson(lam=proj_garf_2e8, size = proj_garf_2e8.shape)
proj_garf_5e8_poisson = np.random.poisson(lam=proj_garf_5e8, size = proj_garf_5e8.shape)

inside = proj_pvfree==296

fig,ax = plt.subplots(4,2)
ax[0,0].imshow(proj_pvfree)
ax[0,1].imshow(proj_mc)
ax[1,0].imshow(proj_garf_5e7)
ax[1,1].imshow(proj_garf_5e7_poisson)
ax[2,0].imshow(proj_garf_5e8)
ax[2,1].imshow(proj_garf_5e8_poisson)
ax[3,0].imshow(proj_garf_1_3e8)
ax[0,0].set_title('PVfree')
ax[0,1].set_title('MC 5e8')
ax[1,0].set_title('garf 5e7')
ax[1,1].set_title('garf 5e7 + POISSON')
ax[2,0].set_title('garf 5e8')
ax[2,1].set_title('garf 5e8 + POISSON')
ax[3,0].set_title('garf 1.3e8')



head = ["mthd", "Counts", "Mean", "Variance"]
data = [["MC", np.sum(proj_mc), np.mean(proj_mc[inside]), np.var(proj_mc[inside])],
        ["garf_5e7", np.sum(proj_garf_5e7), np.mean(proj_garf_5e7[inside]), np.var(proj_garf_5e7[inside])],
        ["garf_5e7_poisson", np.sum(proj_garf_5e7_poisson), np.mean(proj_garf_5e7_poisson[inside]), np.var(proj_garf_5e7_poisson[inside])],
        ["garf_5e8", np.sum(proj_garf_5e8), np.mean(proj_garf_5e8[inside]), np.var(proj_garf_5e8[inside])],
        ["garf_5e8_poisson", np.sum(proj_garf_5e8_poisson), np.mean(proj_garf_5e8_poisson[inside]), np.var(proj_garf_5e8_poisson[inside])],
        ["garf_1.3e8", np.sum(proj_garf_1_3e8), np.mean(proj_garf_1_3e8[inside]), np.var(proj_garf_1_3e8[inside])],]
print(tabulate(tabular_data=data, headers=head))


fig3,ax3 = plt.subplots(6,1)
bins = np.linspace(-1, 22, 24)
bins__ = bins + 0.5
ax3[0].hist(proj_mc[inside], bins = bins__, color = 'green',alpha = 1, label = 'MC 5e8', density = True)
lambdaa = np.mean(proj_mc[inside])
poisson_dist = np.exp(-lambdaa)*np.power(lambdaa, bins)/factorial(bins)
ax3[0].plot(bins, poisson_dist, '-o', color = 'black')

ax3[1].hist(proj_garf_5e7[inside], bins = bins__, color ='red', alpha = 1, label ='garf 5e7 to 5e8', density = True)
ax3[1].plot(bins, poisson_dist, '-o', color = 'black')

ax3[2].hist(proj_garf_5e7_poisson[inside], bins = bins__, color ='blue', alpha = 1, label ='garf 5e7 to 5e8 + POISSON', density = True)
ax3[2].plot(bins, poisson_dist, '-o', color = 'black')

ax3[3].hist(proj_garf_5e8[inside], bins = bins__, color ='orange', alpha = 1, label ='garf 5e8', density = True)
ax3[3].plot(bins, poisson_dist, '-o', color = 'black')

ax3[4].hist(proj_garf_5e8_poisson[inside], bins = bins__, color ='cyan', alpha = 1, label ='garf 5e8 + POISSON', density = True)
ax3[4].plot(bins, poisson_dist, '-o', color = 'black')


ax3[5].hist(proj_garf_1_3e8[inside], bins = bins__, color ='gold', alpha = 1, label ='garf 1.3e8', density = True)
ax3[5].plot(bins, poisson_dist, '-o', color = 'black')


for k in range(len(ax3)):
        ax3[k].legend()
plt.show()