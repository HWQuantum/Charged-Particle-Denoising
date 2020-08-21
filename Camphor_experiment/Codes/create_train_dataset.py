import numpy as np
from scipy import io as sio
import os



### --- TO DEFINE ---
Directory = './Camphor_experiment/' 
### -----------------

save_path = os.path.join(Directory, 'Training_dataset')

## CODE
N = 10000
w = 100
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


# Defines cartesian coordinates
x = range(-100,100,1)
y = range(0,100,1)
x_mat = np.zeros((len(y), len(x)))
y_mat =  np.zeros((len(y), len(x)))
for i in y:
    x_mat[i,:] = x
for j in range(len(x)):
    y_mat[:,j] = y

x = x_mat
y = y_mat

# Defines polar grid with theta = 0 along the x axis
r = np.sqrt((np.square(x) + np.square(y)))
theta = np.arctan2(y,x)

# Defines Legendre polynomials up to order 6

leg_P = np.zeros((100,200,7))
leg_P[:,:,1] = np.ones(r.shape)
leg_P[:,:,2] = np.cos(theta)
for l in range(2,6,1):
    interm_res = np.multiply(leg_P[:,:,2],leg_P[:,:,l])
    leg_P[:,:,l+1] = (1/(l+1))*((2*l+1)*interm_res - l*leg_P[:,:,l-1])
    
#Create blank space for N images
I_slice = np.zeros((100,200,N)) #slice images
I_projected = I_slice # projected images

for n in range(N):
    if np.mod(n,1000) == 0:
        print(n)
    I = np.zeros((100,200))

    #Create distributions
    for i in range(1,5): #Random number of rings between 1 and 5
        R = 90*np.random.rand() # Maximum radius of 90 pixels
        W = (10-1)*np.random.rand() + 1 #Maximum width of 10 pixels
        rad = np.exp(-(np.square(np.divide(r-R,W)))) # Create ring
        asym = 0.25 # Relative asymmetry (images have only ~25% asymmetry at most)   
        B0 = np.random.rand() # Random intensity of ring (0 - +1)   
        B1 = asym*np.random.rand()*(2*np.random.rand() -1) # Random B1 aysmmetry (-0.2 - +0.2)    
        B2 = 3*np.random.rand() - 1 # random B2 (-1 - 2)   
        B3 = asym*(2*np.random.rand()-1) # Random B3 aysmmetry (-0.2 - +0.2)   
        B4 = (2*np.random.rand() - 1) # Random B4 (-1 - +1)   
        B5 = asym*(2*np.random.rand() -1) # Random B5 aysmmetry (-0.2 - +0.2)   
        B6 = (2*np.random.rand() - 1) # Random B4 (-1 - +1)  
        ang = B0*(1 + B1*leg_P[:,:,1] + B2*leg_P[:,:,2]  + B3*leg_P[:,:,3] + B4*leg_P[:,:,4] + B5*B0*leg_P[:,:,5] + B6*leg_P[:,:,6]) # Create angular distribution
    
        # Test if angular distribution is physical 
        if np.amin(ang) >= 0:
            I_working = np.multiply(ang,rad) # Apply to ring
        else:      
            ang = ang - np.amin(ang) # Set minimum to zero      
            I_working = np.multiply(ang,rad) # Apply to ring
        
        I = I + I_working # Add to image
    
    I_slice[:,:,n] = I

    #Project image to simulate VMI conditions
    P = 2*np.dot(A,I)
    I_projected[:,:,n] = P

sio.savemat(os.path.join(save_path,  "IMAGES_PROJECTED_python.mat" ),{'I_projected_PY':I_projected})
