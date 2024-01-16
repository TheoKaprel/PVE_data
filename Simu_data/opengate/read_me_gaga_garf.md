Steps to use both gaga/garf


Prerequisite: 

- create a venv (gagarf_venv)
- pip install torch, Path
- pip install garf
- pip install gaga-phsp
- pip install opengate


- Le model 'garf' de l'intevo est à télécharger dans le repertoire gitlab "https://gitlab.in2p3.fr/opengate/spect_siemens_intevo"
- y'a des data/ct/src/pth dans "https://www.creatis.insa-lyon.fr/~dsarrut/gagaspect_data/data/"



1) Generate the GAGA training data (phsp)
cd /path/to/tests_gaga_garf_de_david

put you ct/source in ./data
change the ct/source def in test203_main0_generate_gaga_training_data_ct.py

run 

    python test203_main0_generate_gaga_training_data_ct.py

The *training* root file is in test203/gaga_training_dataset_ct_large.root


2) Train the gaga

config file in ./data/cg1.json
parameters à tuner maybe : g/d_layers, g/d_dim, z_dim, epochs

Pour lancer l'entrainement : 

    gaga_train test203/gaga_training_dataset_ct_large.root data/cg1.json -o test203/gan_src.pth


3) Train the arf

Use the script : test203_main0_generate_garf_training_data_tc99m_lehr.py

/!\ In this file (lines 35-35-36) I modified the scatter/peak energy windows to [108.5, 129.5] / [129.5, 150.5] to correspond to our IEC experiments





