import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Assign which GPU to run on, "0" or "1"
import tensorflow as tf
import scipy.io as sio
from tensorflow import keras
tf.keras.backend.clear_session()
import numpy as np
import scipy.io as sio
import time
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, concatenate,Flatten,Reshape,Cropping2D
from tensorflow.python.keras.models import Model


#------------------------------------------------------------------------------------------------------------------------
# Dataset Path and Save Paths
#------------------------------------------------------------------------------------------------------------------------

# To EDIT ----
Directory = './Pyrrole_experiment/'
# ------------

intensity_level = 5
background = 0.01
Dataset_path = os.path.join(Directory, 'Training_dataset','IMAGES_PROJECTED_4_fold.mat')
checkpoint_path = os.path.join(Directory, 'Checkpoint','Checkpoint_nopatchy_MainUNet4layers_100200_'+str(intensity_level)+'_'+str(background)+'/cp.ckpt')
Nepochs = 150

#------------------------------------------------------------------------------------------------------------------------
# Load the data - Training (normalised)
#------------------------------------------------------------------------------------------------------------------------
X = sio.loadmat(Dataset_path)
X = X['IMAGES_PROJECTED_PY']
X = np.swapaxes(X, 0, -1)
X = np.swapaxes(X, 1, 2)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
X_noisy = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))

for index in range(X.shape[0]):
    image = np.reshape(X[index,:,:,:], (X.shape[1], X.shape[2]))
    max_image, min_image = np.amax(image), np.amin(image)
    image_init = (image - min_image)/(max_image - min_image)
    I_Poisson = intensity_level*image + background
    I_Poisson = np.array(I_Poisson)
    I_Poisson = I_Poisson.astype(float)
    I_Poisson_noisy = np.random.poisson(I_Poisson)
    max_noisy, min_noisy = np.amax(I_Poisson_noisy), np.amin(I_Poisson_noisy)
    I_Poisson_noisy = (I_Poisson_noisy - min_noisy) / (max_noisy - min_noisy)
    X_noisy[index, :, :, :] = np.reshape(I_Poisson_noisy, (I_Poisson_noisy.shape[0], I_Poisson_noisy.shape[1], 1))
    X[index, :, :, :] = np.reshape(image_init, (image_init.shape[0], image_init.shape[1], 1))

X = X[:10000]
X_noisy = X_noisy[:10000]
X_training = X[1000:]
X_training_noisy = X_noisy[1000:]
X_validation = X[:1000]
X_validation_noisy = X_noisy[:1000]

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


    deconv4 = Conv2DTranspose(512, (w,w), padding='same', activation=tf.nn.relu, strides=(2,2),use_bias=UB)(conv5)
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
start_time = time.time()
history = model.fit(X_training_noisy, X_training, validation_data = (X_validation_noisy, X_validation), epochs = Nepochs, batch_size = 32, callbacks=[cp_callback])
training_time = time.time()-start_time
sio.savemat(os.path.join(Directory, 'Simulated_data', 'history_'+str(intensity_level)+'_'+str(background)+'.mat'), history.history)

#------------------------------------------------------------------------------------------------------------------------
# Test & Save on validation data
#------------------------------------------------------------------------------------------------------------------------
X_val_noisy = X_validation_noisy[0:5]
X_val = X_validation[0:5]
model.load_weights(checkpoint_path)
# Input
input_data = np.zeros((X_val.shape[1]*2,X_val.shape[2]*2,X_val.shape[0]))
Nimages = X_val.shape[0]
for index in range(Nimages):
    val_image = np.squeeze(X_val_noisy[index, :,:,:])
    data_1 = np.squeeze(val_image)
    data_3 = np.flip(data_1, axis=0)
    res = np.concatenate((data_1, data_3), axis = 0) 
    input_data[:,:, index] = np.squeeze(res)

# Ground Truth
GT_data = np.zeros((X_val.shape[1]*2,X_val.shape[2]*2,X_val.shape[0]))
Nimages = X_val.shape[0]
for index in range(Nimages):
    val_image = np.squeeze(X_val[index, :,:,:])
    data_1 = np.squeeze(val_image)
    data_3 = np.flip(data_1, axis=0)
    res = np.concatenate((data_1, data_3), axis = 0) s
    GT_data[:,:, index] = np.squeeze(res)

# Predictions
predict_data = np.zeros((X_val.shape[1]*2,X_val.shape[2]*2,X_val.shape[0]))
Nimages = X_val.shape[0]
for index in range(Nimages):
    val_image = np.squeeze(X_val_noisy[index, :,:,:])
    val_image = np.reshape(val_image, (1, val_image.shape[0],val_image.shape[1],1))
    start_time = time.time()
    predict_image = model.predict(val_image)
    testing_time = time.time() - start_time
    data_1 = np.squeeze(predict_image)
    data_3 = np.flip(data_1, axis=0)
    res = np.concatenate((data_1, data_3), axis = 0) 
    predict_data[:,:, index] = np.squeeze(res)

    
dictionary = {}
dictionary['simulated'] = {'input':input_data, 'prediction':predict_data, 'GT':GT_data}
dictionary['time'] = {'training_time':training_time, 'testing_time_prediction':testing_time}
sio.savemat(os.path.join(Directory, 'Simulated_data', 'res_'+str(intensity_level)+'_'+str(background)+'.mat'), dictionary)




