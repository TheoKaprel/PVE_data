file=$1
basename=${file%.mhd}
rot=${basename}_rot.mhd
rot_2mm=${basename}_rot_2mm.mhd
rot_4mm=${basename}_rot_4mm.mhd

tktk_center_image.py -i ${file} -o ${rot}

clitkAffineTransform -i ${rot} -o ${rot} -m ~/Desktop/PVE/PVE_data/Analytical_data/matrix_rotation.mat --pad 0 --interp 0
echo ${rot}

clitkAffineTransform -i ${rot} -o ${rot_2mm} --spacing 2.3976,2.3976,2.3976 --adaptive --interp 0 
echo ${rot_2mm}

clitkAffineTransform -i ${rot} -o ${rot_4mm}  --spacing 4.6875,4.6875,4.6875 --adaptive --interp 0
echo ${rot_4mm}
