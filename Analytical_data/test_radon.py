import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import itk


def f(x,y):
    return ((x-3)**2)/2**2 + ((y-2)**2)/4**2<1


def radon_f(rho,theta):
    return integrate.quad(lambda t: f(-rho*np.sin(theta)+ t*np.cos(theta),
                                      t*np.sin(theta)+rho*np.cos(theta)),-10,10)[0]

ltheta = np.linspace(0,2*np.pi,100)
lrho = np.linspace(-20,20,100)
drho = lrho[1]-lrho[0]

X = np.zeros((ltheta.shape[0],lrho.shape[0]))
x = np.zeros((lrho.shape[0],lrho.shape[0]))


for i,r in enumerate(lrho):
    for j,s in enumerate(lrho):
        x[i,j] = f(r,s)


xx = itk.image_from_array(x[:,None,:])
xx.SetSpacing([1,1,1])
itk.imwrite(xx, "/export/home/tkaprelian/temp/test_radon/x_src.mhd")

for i,theta in enumerate(ltheta):
    for j,rho in enumerate(lrho):
        pixel_rho = np.linspace(rho-drho/2, rho+drho/2,10)
        for p in pixel_rho:
            X[i,j] += radon_f(p,theta)*drho

XX = itk.image_from_array(X[:,None,:])
XX.SetSpacing([1,1,1])
itk.imwrite(XX, "/export/home/tkaprelian/temp/test_radon/x_ideal.mhd")


alpha = 0.03
sigma0 = 2

def sigma(r):
    return alpha*r+sigma0

def psf(rho,r):
    return 1/np.sqrt(2*np.pi)/sigma(r) * np.exp(-rho**2 / 2 / sigma(r)**2)


r0 = 20
w = lambda delta: psf(r0*np.tan(delta),r0)

Z = np.zeros_like(X)


for i,theta in enumerate(ltheta):
    print(i)
    for j,rho in enumerate(lrho):
        intgrd = lambda d: w(d)*radon_f(r0*d + rho, theta+d)

        abs = np.linspace(-np.pi/15, np.pi/15, 10)
        for k in range(len(abs)-1):
            Z[i,j]+=intgrd((abs[k]+abs[k+1])/2) * (abs[k+1]-abs[k])

ZZ = itk.image_from_array(Z[:,None,:])
ZZ.SetSpacing([1,1,1])
itk.imwrite(ZZ, "/export/home/tkaprelian/temp/test_radon/x_wblur.mhd")


fig,ax = plt.subplots(1,3)
ax[0].imshow(x)
# ax[0].set_xlabel()
ax[1].imshow(X)
ax[2].imshow(Z)
plt.show()