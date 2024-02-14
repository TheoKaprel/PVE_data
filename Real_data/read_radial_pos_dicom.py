#!/usr/bin/env python3

import argparse
import pydicom
import matplotlib.pyplot as plt
import numpy as np

from itk import RTK as rtk

def main():
    print(args)
    ds = pydicom.dcmread(args.input_dicom)
    print(ds)

    radialPosition1,radialPosition2 = [],[]
    radial_pos_detect_1 = ds[0x54, 0x22][0][0x18, 0x1142].value._list
    # print(isinstance(radial_pos_detect_1._list, list))

    radial_pos_detect_1=[radial_pos_detect_1] if (not isinstance(radial_pos_detect_1,list)) else radial_pos_detect_1
    radial_pos_detect_2 = ds[0x54, 0x22][1][0x18, 0x1142].value._list
    radial_pos_detect_2 = [radial_pos_detect_2] if (not isinstance(radial_pos_detect_2, list)) else radial_pos_detect_2

    for v in radial_pos_detect_1:
        radialPosition1.append(float(v))
    for v in radial_pos_detect_2:
        radialPosition2.append(float(v))

    print(f"radial positions 1 : starting at {ds[0x54,0x22][0][0x54, 0x0200].value}")
    print(radialPosition1)
    print(f"radial positions 2 : starting at {ds[0x54,0x22][1][0x54, 0x0200].value}")
    print(radialPosition2)
    print(len(radialPosition1))
    print(len(radialPosition2))


    fig,ax = plt.subplots()
    ax.plot(radialPosition1+radialPosition2)
    # plt.show()

    r1,r2 = radialPosition1,radialPosition2
    theta1 = np.linspace(np.pi, 2*np.pi, 61)[:-1]
    theta2 = np.linspace(0,np.pi, 61)[:-1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta1, r1, c = "blue")
    ax.plot(theta2, r2, c = 'orange')
    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')

    if args.save_geom is not None:
        geometry = rtk.ThreeDCircularProjectionGeometry.New()
        radial_distances = radialPosition2 + radialPosition1
        angles = np.linspace(0, 360, len(radial_distances)+1)
        for i in range(len(radial_distances)):
            geometry.AddProjection(radial_distances[i], 0, angles[i], 0, 0)

        geom_file_writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
        geom_file_writer.SetObject(geometry)
        geom_file_writer.SetFilename(args.save_geom)
        geom_file_writer.WriteFile()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dicom")
    parser.add_argument("--save_geom", type = str, default=None)
    args = parser.parse_args()

    main()
