import uproot
import numpy as np


with uproot.open('outputs/arf/arf_training_dataset.root') as f:
    k = f.keys()
    print(k)

    arf = f['ARF_training']

    print(arf)

    # Get keys
    names = [k for k in arf.keys()]

    print(names)

    n = arf.num_entries

    print(n)

    a = arf.arrays(library="numpy")

    d = np.column_stack([a[k] for k in arf.keys()])
    # print(d)

    # print(d[d[:,3]>0])
    print(d[d[:,3]>0].shape[0])
    # print(np.mean(d[d[:,3]>0][:,2]))

    # print((d[d[:,3]==1][:,0]<=0.126).all() and (d[d[:,3]==1][:,0]>=0.114).all())
    # print((d[d[:,3]==2][:,0]>0.126).all() and (d[d[:,3]==2][:,0]<0.154).all())

