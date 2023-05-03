#!/usr/bin/env python3

import argparse
import pydicom
import matplotlib.pyplot as plt

def main():
    print(args)
    ds = pydicom.dcmread(args.input_dicom)
    print(ds)

    radialPosition1,radialPosition2 = [],[]
    for v in ds[0x54, 0x22][0][0x18, 0x1142].value:
        radialPosition1.append(float(v))
    for v in ds[0x54, 0x22][1][0x18, 0x1142].value:
        radialPosition2.append(float(v))

    print("radial positions 1 : ")
    print(radialPosition1)
    print("radial positions 2 : ")
    print(radialPosition2)
    print(len(radialPosition1))
    print(len(radialPosition2))


    fig,ax = plt.subplots()
    ax.plot(radialPosition1+radialPosition2)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dicom")
    args = parser.parse_args()

    main()
