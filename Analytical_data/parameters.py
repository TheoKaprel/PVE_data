import numpy as np


# Units in mm

# holes diameter
d = 1.5

# holes length
l = 35

# mass attenuation coefficient at 150 keV in Pb
mu = 1.91*11.35*10

# Effective length
leff = l - 2/mu

# Distance center of FOV / detector's head
b = 380

# OLD PARAMETERS
# sigma0pve_default = 0.9008418065898374
# alphapve_default = 0.025745123547513887


sigma0pve_default = d / (2*np.sqrt(2*np.log(2)))
alphapve_default = d / (2*np.sqrt(2*np.log(2))) / leff
