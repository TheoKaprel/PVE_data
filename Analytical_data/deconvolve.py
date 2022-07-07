import numpy as np
import matplotlib.pyplot as plt
from skimage import color, restoration
import itk


img1PVE = itk.array_from_image(itk.imread('./test_deconvolution/src1proj_PVE.mhd'))[0,:,:]
img1PVf = itk.array_from_image(itk.imread('./test_deconvolution/src1proj_PVfree.mhd'))[0,:,:]
psf1 = restoration.richardson_lucy(img1PVE, img1PVf, filter_epsilon=0.00001)

img1DeepPVC = itk.array_from_image(itk.imread('./test_deconvolution/src1proj_DeepPVC.mhd'))[0,:,:]
psf1Deep= restoration.richardson_lucy(img1PVE, img1DeepPVC, filter_epsilon=0.00001)


img2PVE = itk.array_from_image(itk.imread('./test_deconvolution/src2proj_PVE.mhd'))[0,:,:]
img2PVf = itk.array_from_image(itk.imread('./test_deconvolution/src2proj_PVfree.mhd'))[0,:,:]
psf2 = restoration.richardson_lucy(img2PVE,img2PVf, filter_epsilon=0.00001)

img2DeepPVC = itk.array_from_image(itk.imread('./test_deconvolution/src2proj_DeepPVC.mhd'))[0,:,:]
psf2Deep= restoration.richardson_lucy(img2PVE, img2DeepPVC, filter_epsilon=0.00001)


vmin_img = min((np.min(img1PVE), np.min(img1PVf), np.min(img2PVE), np.min(img2PVf), np.min(img1DeepPVC), np.min(img2DeepPVC)))
vmax_img = max((np.max(img1PVE), np.max(img1PVf), np.max(img2PVE), np.max(img2PVf), np.max(img1DeepPVC), np.max(img2DeepPVC)))
vmin_psf = min((np.min(psf1), np.min(psf2), np.min(psf1Deep), np.min(psf2Deep)))
vmax_psf = max((np.max(psf1), np.max(psf2), np.max(psf1Deep), np.max(psf2Deep)))


fig, ax = plt.subplots(2,5)
ax[0,0].imshow(img1PVE, vmin=vmin_img, vmax=vmax_img)
ax[0,0].set_title('Proj_PVE')
ax[0,0].set_ylabel('Source at distance \n d = 9 cm  \n to the detector', fontsize = 18, rotation = 0, labelpad=60)
ax[0,1].imshow(img1PVf, vmin=vmin_img, vmax=vmax_img)
ax[0,1].set_title('Proj_PVfree')
ax[0,2].imshow(psf1, vmin=vmin_psf, vmax = vmax_psf)
ax[0,2].set_title('deconvolution h tel que \n Proj_PVE = Proj_PVfree * h')
ax[0,3].imshow(img1DeepPVC, vmin = vmin_img, vmax=vmax_img)
ax[0,3].set_title('Proj_DeepPVC')
ax[0,4].imshow(psf1Deep, vmin = vmin_psf, vmax = vmax_psf)
ax[0,4].set_title('deconvolution h tel que \n Proj_PVE = Proj_DeepPVC * h')
ax[1,0].imshow(img2PVE, vmin=vmin_img, vmax=vmax_img)
ax[1,0].set_title('Proj_PVE')

ax[1,0].set_ylabel('Source at distance \n d = 29 cm  \n to the detector', fontsize = 18, rotation = 0, labelpad=60)
ax[1,1].imshow(img2PVf, vmin=vmin_img, vmax=vmax_img)
ax[1,1].set_title('Proj_PVfree')
ax[1,2].imshow(psf2, vmin = vmin_psf, vmax=vmax_psf)
ax[1,2].set_title('deconvolution h tel que \n Proj_PVE = Proj_PVfree * h')
ax[1,3].imshow(img2DeepPVC, vmin = vmin_img, vmax=vmax_img)
ax[1,3].set_title('Proj_DeepPVC')
ax[1,4].imshow(psf2Deep, vmin = vmin_psf, vmax = vmax_psf)
ax[1,4].set_title('deconvolution h tel que \n Proj_PVE = Proj_DeepPVC * h')

plt.show()