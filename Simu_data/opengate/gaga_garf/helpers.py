import scipy
import numpy as np

def update_ideal_recons(batch,recons,offset,spacing,size,e_min=0.001):
    batch=batch.cpu().numpy()
    c = scipy.constants.speed_of_light * 1000  # in mm

    # loop on x ; check energy
    positions = batch[:,1:4]
    directions = batch[:,4:7]

    times = batch[:,7] / 1e9
    energies = batch[:,0]

    # filter according to E ?
    # mask = energies > e_min
    mask = (energies>e_min*0.99) & (energies<e_min*1.01)
    positions = positions[mask]
    directions = directions[mask]
    times = times[mask]

    # output
    emissions = np.zeros_like(positions)

    for pos, dir, t, E, p in zip(positions, directions, times, energies, emissions):
        l = t * c
        p += pos + l * -dir

    pix = np.rint((emissions - offset) / spacing).astype(int)

    size=[size[2],size[1],size[0]]
    for i in [0, 1, 2]:
        pix = pix[(pix[:, i] < size[i]) & (pix[:, i] > 0)]

    for x in pix:
        recons[x[2], x[1], x[0]] += 1

    return recons