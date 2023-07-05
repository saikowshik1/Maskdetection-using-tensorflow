# -*- coding: utf-8 -*-
"""
@author: Rithikreddy
"""

#import tensorflow as tf

''' Remember The caps is compulsory while importing '''


from tensorflow.keras.preprocessing.image import ImageDataGenerator      #Genearte batches of tensor image data with data-augmentation
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import load_img   
from tensorflow.keras.applications import MobileNetV2       #mobilenetv2 config is imported
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input        #for preprocessing the input based on the model
from tensorflow.keras.layers import AveragePooling2D        #for the pooling layers
#from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten                 #for flattening the image
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam                #for 'Adam' optimisation algorithm 
from tensorflow.keras.utils import to_categorical           #for converting into one-hot vector
from sklearn.model_selection import train_test_split        #for segmenting dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
#import argparse
import os

#ap = argparse.ArgumentParser()
#ap.add_argument("-d","--dataset",required="True", default="dataset/with_mask",help = "path to input dataset")
#ap.add_argument("-p","--plot",default = "plot.png",help="path to the output plot")
#ap.add_argument("-m","--model",default="mask_detector.model",help="path to output face mask detection model")
#args = vars(ap.parse_args())   
# This is to converting every parsed command line argument into a dictionary with key as name of the argument
# and the value as the given value
#print(args)


INIT_LR = 1e-4    #Initial Learning rate
EPOCHS = 10       
BS = 32           #Batch size


imagepaths = list(paths.list_images("H:/Documents/Data science Lab/mask_detector/dataset"))

#imagepaths = (paths.list_images("H:/Documents/Data science Lab/mask_detector/dataset"))
#print(type(imagepaths))  It is of type class generator now we have to convert it into a list
#print(imagepaths)    #This prints the path of the all the files of type image in all the subdirectories

data = []
labels = []

for imagepath in imagepaths:
    label = imagepath.split(os.path.sep)[-2]         #a.split(x) splits the file 'a' based on the x and os.path.sep is '/'
    #print(label)                                    #It contains the label with_mask and without_mask
    image=load_img(imagepath,target_size=(224,224))  #loading the image from the imagepath and resizing the size of the image to 224,224
    image=img_to_array(image)                        #making the image as 224,224,3 if RGB image 224,224,1 if grey scale image
    #print(image)
    image=preprocess_input(image)                    #preprocessing the image as per the model 
    
    data.append(image)
    labels.append(label)
    
data = np.array(data,dtype="float32")                #dtype is in quotes""
labels = np.array(labels)

lb = LabelBinarizer()
#print(labels)
labels = lb.fit_transform(labels)                   #Make all the labels as 0's and 1's from the strings of woithhout_mask and with_mask
#print(labels)
labels = to_categorical(labels)             #Converts into a one hot vector,depending on no of classes it's shape is decided dynamically
#print(labels)

(trainX,testX,trainY,testY) = train_test_split(data,labels,stratify = labels,test_size=0.2,random_state=42) 
# stratified data is split in a stratified fashion, using this as the class labels.

aug = ImageDataGenerator(
	rotation_range=20,       #It randomly rotates the image clockwise by an angle <= specified value
	zoom_range=0.15,         #It zooms the image uniformly randomly from [1-0.15,1+0.15] remember <1 zooms in >1 zooms out
	width_shift_range=0.2,    #0.2 is 20%percentage in the shift required in the width
	height_shift_range=0.2,   #0.2 is 20%percentage in the shift required in the height
	shear_range=0.15,
	horizontal_flip=True,   #Whether a flip is required in the image or not
	fill_mode="nearest")
#Image DataGeneartor(for data augumentation)


basemodel = MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))  #transfer learning

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel = Flatten(name="flatten")(headmodel)
headmodel = Dense(128,activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2, activation="softmax")(headmodel)


#Placing the headmodel on the top of the basemodel
model = Model(inputs=basemodel.input,outputs=headmodel)

for layer in basemodel.layers:
    layer.trainable = False     #remember f in false is in caps

opt = Adam(learning_rate = INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer = opt,metrics = ["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),      #
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)                                #Training the model
    
predIdxs = model.predict(testX, batch_size=BS)    #Testing the model

predIdxs = np.argmax(predIdxs, axis=1)    #we have to find the index of each image having highest probability

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("loss_plot")











