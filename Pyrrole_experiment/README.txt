-----------------------------------------------------------------------------
An Artificial Neural Network for Noise Removal in Data Sparse Charged Particle Imaging Experiments               
-----------------------------------------------------------------------------
Institute of Photonics & Quantum Sciences, Heriot-Watt University, Edinburgh, EH14 4AS, UK
Contact: d.townsend@hw.ac.uk.


This folder contains the code implementation and trained models used for the
Pyrrole Experiment of the Project. The following describes how to generate training data and perform the training and evaluation on the captured results. We provide data and checkpoints at https://researchportal.hw.ac.uk/en/datasets/dataset-for-artificial-neural-networks-for-noise-removal-in-data-.


==Generating Training Data==
Steps
1. Edit Directory in ./Codes/create_train_dataset.py. Directory is the path to the folder Pyrrole_experiment on your computer. 
2. Run './Codes/create_train_dataset.py'

Commentary
This script is simulating smooth two photon images of our experiment. These correspond to the ground truth images that the network will learn to reconstruct from noisy images. (The noisy images will be simulated in the following training step.)


==Training==
Steps
1. Install required python packages listed in './requirements.txt' 
2. Edit Directory in './Codes/train.py'. Directory is the path to the folder Pyrrole_experiment on your computer. 
3. Edit noise parameters (intensity and background level)
3. Run './Codes/train.py'

Commentary

The code allows training of the network for the Poisson noise defined by the variables intensity_level and background. You can edit those parameters in './Codes/train.py' and run the training to fit your data. 

Results of the training on Simulated Dataset are saved in ./Simulated_Data/res_*intensity_level*_*background_level*.mat. 

Results of the training and validation losses and accuracies are saved in ./Simulated_Data/history_*intensity_level*_*background_level*.mat

Python packages used for the training are listed in the './requirements.txt' file. 



==Evaluation on Captured Dataset==

Steps:

- Either use the already trained network
	1. Edit Directory in ./Codes/evaluate_real.py. Directory is the path to the folder Pyrrole_experiment on your computer. 
	2.Run ./Codes/evaluate_real.py

- Do the training again with noise parameters corresponding to our real data 
	1. Edit Directory in ./Codes/create_train_dataset.py. Directory is the path to the 	folder Pyrrole_experiment on your computer. 
	2. Run './Codes/create_train_dataset.py'
	3. Edit noise parameters (intensity=5 and background level=0.01)
	4. Run ./Codes/train.py
	5. Run ./Codes/evaluate_real.py

Commentary
(We provide the checkpoint of the already trained network in './Checkpoint/' corresponding to the noise parameters : intensity_level=5 and background=0.01.)

The script runs the network on our real data saved in './Real_Data/Data/Pyrrole_Data.mat'. The results are saved in '.Real_Data/Results/res.mat'. 

