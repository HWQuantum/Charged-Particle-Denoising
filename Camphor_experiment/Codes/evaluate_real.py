import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Assign which GPU to run on, "0" or "1"
import tensorflow as tf
import scipy.io as sio
from tensorflow import keras
tf.keras.backend.clear_session()
import numpy as np
import scipy.io as sio
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, concatenate,Flatten,Reshape,Cropping2D
from tensorflow.python.keras.models import Model

#------------------------------------------------------------------------------------------------------------------------
# To define
#------------------------------------------------------------------------------------------------------------------------
intensity_level = 60
background = 0.1
Directory = './Camphor_experiment/'
checkpoint_path = os.path.join(Directory, 'Checkpoint','Checkpoint_model6layers_'+str(intensity_level)+'_'+str(background)+'/cp.ckpt')
Real_data_path = os.path.join(Directory, 'Real_data/Data')

#------------------------------------------------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------------------------------------------------
def create_model():
    UB = True
    w = 5
    inputs = Input((100,200,1))
    conv1 = Conv2D(64, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(inputs)
    pool1 = MaxPool2D((2,2), padding='same')(conv1)

    conv2 = Conv2D(128, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool1)
    pool2 = MaxPool2D((2,2), padding='same')(conv2)

    conv3 = Conv2D(256, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool2)
    pool3 = MaxPool2D((2,2), padding='same')(conv3)

    conv4 = Conv2D(512, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool3)
    pool4 = MaxPool2D((2,2), padding='same')(conv4)

    conv5 = Conv2D(1024, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool4)
    pool5 = MaxPool2D((2,2), padding='same')(conv5)
    
    conv6 = Conv2D(2048, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool5)
    pool6 = MaxPool2D((2,2), padding='same')(conv6)

    conv7 = Conv2D(4096, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(pool6)
   

    deconv6 = Conv2DTranspose(2048, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(conv7)
    cropped_decon6 = Cropping2D(cropping=((0, 0), (1, 0)))(deconv6)
    merge6 = concatenate([conv6,cropped_decon6], axis = 3)

    deconv5 =  Conv2DTranspose(1024, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge6)
    cropped_decon5 = Cropping2D(cropping=((1, 0), (1, 0)))(deconv5)
    merge5 = concatenate([conv5,cropped_decon5], axis = 3)
    deconv4 = Conv2DTranspose(512, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge5)
    cropped_decon4 = Cropping2D(cropping=((1, 0), (1, 0)))(deconv4)
    merge4 = concatenate([conv4,cropped_decon4], axis = 3)

    deconv3 = Conv2DTranspose(256, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge4)
    cropped_deconv3 = Cropping2D(cropping=((1, 0), (0, 0)))(deconv3)
    merge3 = concatenate([conv3,cropped_deconv3], axis = 3)

    deconv2 = Conv2DTranspose(128, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge3)
    
    merge2 = concatenate([conv2,deconv2], axis = 3)
    deconv1 = Conv2DTranspose(64, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(merge2)
    merge1 = concatenate([conv1,deconv1], axis = 3)
    conv_last = Conv2D(1, (w,w), padding='same', activation=tf.nn.relu,use_bias=UB)(merge1)

    model = Model(inputs,conv_last)

    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.00010), loss = "mean_squared_error", metrics = ["accuracy"])
    
    return model

model = create_model()

model.summary()


checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.load_weights(checkpoint_path)

# #------------------------------------------------------------------------------------------------------------------------
# # Test & Save on real data
# #------------------------------------------------------------------------------------------------------------------------

data = sio.loadmat('/home/ar432/VMI_Project/Dataset/Chris_2_fold/S_Camphor_All_Shots.mat')
LCP_real = np.double(data['LCP_shots'])
X = LCP_real
X = X[200:400:2,0:400:2,:]
X_new = X
for index in range(X.shape[2]):
    image = X[:,:,index]
    min_val, max_val = np.amin(image), np.amax(image)
    image = (image - min_val) / (max_val - min_val)
    X_new[:,:,index] = image
X = X_new

X = np.swapaxes(X, 0, -1)
X = np.swapaxes(X, 1, 2)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
res = model.predict(X)

X_1 = X
X_2 = np.flip(X,1)
X = np.concatenate((X_2,X_1),axis = 1)

res_1 = res
res_2 = np.flip(res,1)
res = np.concatenate((res_2,res_1),axis = 1)
dictionary = {}
dictionary['real_LCP'] = {'input':X, 'prediction':res}


RCP_real = np.double(data['RCP_shots'])
X = RCP_real
X = X[200:400:2,0:400:2,:]
X_new = X
for index in range(X.shape[2]):
    image = X[:,:,index]
    min_val, max_val = np.amin(image), np.amax(image)
    image = (image - min_val) / (max_val - min_val)
    X_new[:,:,index] = image
X = X_new
X = np.swapaxes(X, 0, -1)
X = np.swapaxes(X, 1, 2)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
res = model.predict(X)

X_1 = X
X_2 = np.flip(X,1)
X = np.concatenate((X_2,X_1),axis = 1)

res_1 = res
res_2 = np.flip(res,1)
res = np.concatenate((res_2,res_1),axis = 1)
dictionary['real_RCP'] = {'input':X, 'prediction':res}


sio.savemat(os.path.join(Directory, 'Real_Data', 'Results', 'res_'+str(intensity_level)+'_'+str(background)+'.mat'), dictionary)
