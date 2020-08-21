-----------------------------------------------------------------------------
An Artificial Neural Network for Noise Removal in Data Sparse Charged Particle Imaging Experiments               |
-----------------------------------------------------------------------------
Institute of Photonics & Quantum Sciences, Heriot-Watt University, Edinburgh, EH14 4AS, UK
Contact: d.townsend@hw.ac.uk.

This folder contains the code implementation and trained models used for the
project. The following describes how to generate training data and perform the
training and evaluation on the captured results. Packaged with the code we include our real data.


==Generating Training Data==
Steps
1. Edit Directory in ./Codes/create_train_dataset.py. Directory should be the path to the folder where Camphor_experiment is.
1. Run ./Codes/create_train_dataset.py

Commentary
This script is simulating smooth two photon images of our experiment. These correspond to the ground truth images that the network will learn to reconstruct from noisy images. (The noisy images will be simulated in the following training step.)


==Training==
Steps
1. Install required python packages listed in './requirements.txt' 
2. Edit the name of the Directory in ./Codes/train.py. Directory should be the path to the folder where Camphor_experiment is.
3. Edit noise parameters (intensity and background level)
3. Run ./Codes/train.py

Commentary
The code allows training of the network for the Poisson noise defined by the variables intensity_level and background. You can edit those parameters in './Codes/train.py' and run the training to fit your data. 
Results of the training on Simulated Dataset are saved in ./Simulated_Data/res_*intensity_level*_*background_level*.mat
Python packages used for the training are listed in the './requirements.txt' file. 



==Evaluation on Captured Dataset==

Steps:
- Either use the already trained network
	1. Edit the name of the Directory in ./Codes/evaluate_real.py. Directory should be the path to the folder where Camphor_experiment is.
	2. Run ./Codes/evaluate_real.py

- Do the training again with noise parameters corresponding to our real data 
	1. Edit the name of the Directory in ./Codes/create_train_dataset.py. Directory should be the path to the folder where Camphor_experiment is.
	2. Run ./Codes/create_train_dataset.py
	3. Edit noise parameters (intensity=5 and background level=0.001)
	4. Run ./Codes/train.py
	5. Run ./Codes/evaluate_real.py

Commentary
We provide the checkpoint of the already trained network in './Checkpoint/' corresponding to the noise parameters : intensity=5 and background=0.001.
The script runs the network on our real data saved in './Real_Data/Data/S_Camphor_All_Shots.mat' for both LCP and RCP polarised light. The results are saved in '.Real_Data/Results/res_*intensity_level*_*background_level*.mat'. 

