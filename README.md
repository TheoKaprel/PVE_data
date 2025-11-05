# Generate SPECT Analytical Data 

Example script to generate 10 random samples:


    python Analytical_data/generate_random_dataset.py -n 10 --type mha --dtype float32 --min_radius 4 --max_radius 32 --min_ratio 1 --max_ratio 60 --min_activity 100 --max_activity 1000 --acquisition_time 15 --radionuclide Lu177 --nspheres 8 --background --sphere 0.7 --ellipse 0.1 --cylinder 0.1 --convex 0.1 --grad_act --nproj 120 --sid 280 --fov 532,388 --attenuationmapfolder attmap_data/ --organlabels rois_labels.json --organratios activity_ratios.json --organproba 0.5 --spect_system siemens-intevo-megp --save_src --rec_fp 10 --project --output_folder output_folder

I know this is not optimal but the --attenuatiomapfolder must contain the following files, for each CT image: 
* XXX_attmap_cropped_rot.mhd: the attenution map at the CT resolution in the desired orientation.
* XXX_rois_labels_cropped_rot.mhd: the corresponding ROIs label mask (still at the CT resolution).
* XXX_attmap_cropped_rot_4mm.mhd: the attenuation map with a 4mm resolution, which will be used for reconstructions (rec_fp)

The --organlabels json file specifies the labels contained in the ROIs label masks (XXX_rois_labels_cropped_rot.mhd images). It should be of the following form:

    {
    "body": 1,
    "liver": 2,
    "kidney_left": 3,
    "kidney_right": 4,
    "spleen": 5,
    "gallbladder": 6,
    "stomach": 7,
    "pancreas": 8,
    "small_bowel": 9,
    "colon": 10,
    "duodenum": 11,
    "urinary_bladder": 12
    }


The --organratios json file should contain [minTBR, maxTBR] values, in the following form:

    {
    "body": [1,1],
    "liver": [6,20],
    "kidney_left": [20,60],
    "kidney_right": [20,60],
    "spleen": [20,60],
    "gallbladder": [2,8],
    "stomach": [2,8],
    "pancreas": [2,8],
    "small_bowel": [2,8],
    "colon": [2,8],
    "duodenum": [2,8],
    "urinary_bladder": [6,20]
    }


