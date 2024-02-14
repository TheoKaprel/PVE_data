# Notes on "Learning SPECT detector angular response function with neural network for accelerating Monte-Carlo simulations" (Sarrut et al)


### intro
ARF = replacing explicit photon tracking with a tabulated model of the Collimator-Detector Response Function (CDRF)

The look up table is dertermined by a simulation 
Then, the model takes as input deirection angles, energy of incoming photons.
Total speed-up factor (tracking in CT + SPECT head) \in [20, 100] but not estimated just for detection part yer

Main limitation: 10^9 - 10^11 primary photons needed to have a good statistical uncertainty to determine the tabulated function. And this for each Energy Window

### Method

Replace ARF tabs by NN

#### 1. Source

Plane in front of the collimator. \
Photons energy uniformly distributed between [E_min, E_max], with E_max= largest energy of the Radionuclide\
Isotropic directions


#### 2. Detection

For each photon reaching the detector, we record:
- incident angles theta, phi
- incident Energy E
- label of energy window in which the photon was detected (including 0 or W_ND, the EW for 'non-detected photons')

#### 3. Russian Roulette

Most photons end up in the W_ND window (99%). In order to re-balance the nb of counts in each EW, we store only 1/w counts of the W_ND window. 
Typicaly w = 40 or 100. This RR factor is taken into account later for prediction step



#### 4. Training

We have n EW in the training dataset

input: x = (theta, phi, E)
output to predict: y \in R^n vector = "one-hot" vector of the proba that the photon is deteced in each EW

h(theta, phi, E) = (y_0, y_1, ... , y_{n-1})

NN architecture: 
- 3 fully connected hidden linear layers with H=400 neurons. 
- Activation function= ReLu
- Loss function: multiclass cross entropy



#### 5. Image Generation

Simulation is run with SPECT head replaced by empty plane of 1nm thickness (ARF plane) located in front of the collimator 
Position, direction and energy of photons that reach the ARF plane are stored. Then The image is compted as follows: 
For each photon, the coordinate (u,v) of the photon in the image plane (sampled with 4x4mm pixel of other). With values (theta, phi, E) of each incident photon, the probabilities that the photon is detected in each EW are computed by h(theta, phi, E) = y
Then , 

I(u,v,i) += y_i

where i is the index of the energy window

The nn-ARF model (as well as tabulated ARF) outputs a sum of probabilities in all energy windows

#### Results

###### 

###### NOISE

"Regarding noise analysis: the analog Monte-Carlo simulation reproduced the Poisson
noise properties, with the distribution’s mean value equal to its variance for the three tested numbers of primary
particles: 5 × 10 8 , 10 9 and 2 × 10 9. Fitted Poisson curves are also shown in the figure. For the ARF method, the mean counts were very close to the analog mean counts (less than 1% difference), but the distributions were not
Poisson. They might be approximated by a Gaussian distribution (black dashed line in the figure). In this case,
ARF distributions were obtained for 5 × 10 7 particles and scaled to the number of particles of the analog simulation. If realistic noise is needed, the ARF image should be scaled to the expected number of primary particles and
pixel values perturbed by stochastic samples drawn from a Poisson distribution with mean equal to the current
pixel count."









