import uproot
import numpy as np


with uproot.open('outputs/arf/arf_5x10_9.root') as f:
    k = f.keys()
    print(k)

    arf = f['ARF_training']

    print(arf)

    # Get keys
    names = [k for k in arf.keys()]

    print(names)

    n = arf.num_entries

    print(f'Total number of particles : {n}')

    a = arf.arrays(library="numpy")

    d = np.column_stack([a[k] for k in arf.keys()])
    # print(d)
    print(d[34,:])

    # print(d[d[:,3]>0])
    nEW = d[d[:,3]>0].shape[0]
    print(f'Number of particles in one of the Energy Windows : {nEW}')



    # print(np.mean(d[d[:,3]>0][:,2]))

    # print((d[d[:,3]==1][:,0]<=0.126).all() and (d[d[:,3]==1][:,0]>=0.114).all())
    # print((d[d[:,3]==2][:,0]>0.126).all() and (d[d[:,3]==2][:,0]<0.154).all())

