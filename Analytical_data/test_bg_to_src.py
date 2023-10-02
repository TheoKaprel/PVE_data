import numpy as np
import matplotlib.pyplot as plt


N = 50000

m,M = 1/1000,1/8

R = 100
lbda = np.log(R)/(M-m)
S = 1/(M+np.log(np.random.rand(N))/lbda)
S[1/S<m] = 1/m


fig,ax = plt.subplots()
ax.hist(1/S,bins = 200)
plt.show()


N = 50000

m,M = 1/1000,1/8

R = 100
lbda = np.log(R)/(M-m)
S = 1/(M+np.log(np.random.rand(N))/lbda)
S[1/S<m] = 1/m


fig,ax = plt.subplots()
ax.hist(1/S,bins = 200)
plt.show()