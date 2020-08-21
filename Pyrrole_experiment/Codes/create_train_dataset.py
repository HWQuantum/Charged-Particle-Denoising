# ------------------------------------------------------------------------------------
# RANDOM VMI IMAGE GENERATION FOR TRAINING DATASET of NETWORK

# Produces a series of random VMI images to use for training the Leach groups neural network

# Generates N random VMI images of size SxS

# Define random image constraints
# S = Each image is SxS pixels
# N = number of random images
# k_max = max possible no of Gaussians
# R_max = max possible radius of Gaussian
# w_max = max possible SD of Gaussian

import numpy as np
from scipy import io as sio
import os
from numpy.random import choice

### -------------- TO DEFINE --------------
Directory = './Pyrrole_Experiment/'
### -----------------------------------------

save_path = os.path.join(Directory, 'Training_dataset')
S = 200
N = 10000
k_max = 10
R_max = S/2
w_max = 20
w = int(S/2)

# -------------- Import B2_vector, B4 and B2 --------------  

Dir = os.path.join(Directory, 'Coef_training_data')
B2_vector = sio.loadmat(os.path.join(Dir,'B2_vector_mat.mat'))
B2_vector = np.squeeze(B2_vector['B2_vector'])

B4_upper = sio.loadmat(os.path.join(Dir,'B4_upper_mat.mat'))
B4_upper = np.squeeze(B4_upper['B4_upper'])

B4_lower = sio.loadmat(os.path.join(Dir,'B4_lower_mat.mat'))
B4_lower = np.squeeze(B4_lower['B4_lower'])


IMAGES_SLICE = np.zeros((S,S,N))
IMAGES_PROJECTED = np.zeros((S,S,N))

# Create coordinate variables
X = np.zeros((S,S))

for n in range(S):
    X[n,:] = range(-int(S/2), int(S/2) , 1)

Y = np.transpose(X)
R = np.sqrt(np.square(X) + np.square(Y))
T = np.transpose(np.arctan2(Y,X))  
cos_T = np.cos(T)
P2 = 0.5*(3*np.square(cos_T) - 1)
P4 = (1/8)*(35*np.square(np.square(cos_T)) - 30*np.square(cos_T) + 3)

# Abel transform Matrix
w = int(S/2)
A = np.zeros((w,w))
for r in range(1, w+1 , 1):         #for each radius
    for y in range(1, w+1 , 1):     #for each slice of y
        if y==r:                    #if your touching the y axis
            A[r-1,y-1] = -0.5*np.sqrt(-1+2*r)*r + 0.5*np.sqrt(-1+2*r) - 0.5*np.square(r)*np.arcsin((r-1)/r) + 0.25*np.square(r)*np.pi
        elif y<r:                   #if below diagonal do nothing
            A[r-1, y-1] = 0      
        else:                       #if above diagonal 
            A[r-1,y-1]=(-0.5 * np.square(y) + y - 0.5) * np.arcsin(r/(y-1)) + (0.5 * np.square(y) - y + 0.5)*np.arcsin((r-1)/(y-1)) \
                + 0.5 * np.sqrt( np.square(y) - 2*y + 2*r - np.square(r))*r - 0.5 * np.sqrt(np.square(y) + 2*r - 1 - np.square(r)) * r - 0.5 * np.sqrt( np.square(y) - 2*y + 2*r - np.square(r))  \
                + 0.5 * np.sqrt(np.square(y) + 2*r - 1 - np.square(r)) + 0.5 * np.sqrt( np.square(y) - np.square(r)) * r - 0.5 * np.square(y)*np.arcsin((r-1)/y)  \
                + 0.5 * np.square(y) * np.arcsin(r/y) - 0.5*np.sqrt( np.square(y) - 2*y + 1 - np.square(r)) * r

# -------------- GENERATE IMAGES --------------   
#  Generates N images consisting of 1:k_max Gaussian rings of radius 0:R_max and width 1:w_max.
#  Each ring has a random intensity (B0) and a random anisotropy (B2).

for n in range(N):
    if np.mod(n, 1000)==0:
        print(n)

    I_SLICE = np.zeros((S,S))
    I_PROJECTED = np.zeros((S,S))
    k = np.random.randint(1, k_max+2) -1 

    for j in range(k+1):
        R0 = np.random.randint(1,R_max+1) 
        w0 = np.random.randint(1,w_max+1)
        B0 = np.random.rand()
        B2 = -1 + 3*np.random.rand()
        idx = np.argmin(np.abs(B2_vector - B2))
        B4 = B4_lower[idx] + (B4_upper[idx]-B4_lower[idx])*np.random.rand()

        I_SLICE = I_SLICE + np.exp(-np.square((R-R0)/w0)) * B0 * (1 + B2*P2 + B4*P4)
        

    I_SLICE = I_SLICE / np.amax(I_SLICE) # normalise

    IMAGES_SLICE[:,:,n] = I_SLICE 

    I_SLICE = np.transpose(I_SLICE)

    I_PROJECTED[range(w , 0 , -1),:] = 2 * np.matmul(A,I_SLICE[range(w,0,-1),:])
    I_PROJECTED[range(w , 2*w ), :] = I_PROJECTED[range(w,0,-1), :]


    I_PROJECTED = I_PROJECTED/np.amax(I_PROJECTED) # normalise
    
    IMAGES_PROJECTED[:,:,n] = np.transpose(I_PROJECTED) 


print(IMAGES_PROJECTED.shape)
IMAGES_PROJECTED = IMAGES_PROJECTED[:100, :200,:]

# -------------- SAVE IMAGES --------------  

sio.savemat(os.path.join(save_path,  "IMAGES_PROJECTED_4_fold.mat" ),{'IMAGES_PROJECTED_PY':IMAGES_PROJECTED})
print('Done!')

