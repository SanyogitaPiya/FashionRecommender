import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
import cv2
import numpy as np
from numpy.linalg import norm
import os
import pickle

# This function runs pretrained model ResNet50 with one layer added
def load_pretrained():
    #we are using pretrained model of resnet50 which is trained on imagenet dataset
    pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # We add GlobalAveragePooling2D layer to the ResNet50 model for fixed-length vector that can be fed into the subsequent layers of CNN
    # There are 2 choices we can make GlobalAveragePooling2D or GlobalMaxPooling2D
    x = pretrained.output
    x = GlobalAveragePooling2D()(x)
    # Create a new model with the ResNet50 base model and the GlobalAveragePooling2D layer
    model = tensorflow.keras.Model(inputs=pretrained.input, outputs=x)
    return model

# This function takes one image filename and the CNN model and gives the feature values of that image
def feature_extraction(filename,model):
    #img = image.load_img(filename)
    img = cv2.imread(filename)
    img = cv2.resize(img,(224,224))
    #convert image to np array
    img = np.array(img)
    #pre-trained CNN was trained on a dataset with a specific input shape so expand shape
    img1 = np.expand_dims(img, axis=0)
    #This step is needed for images when using resnet50. This is a built in function call
    process = preprocess_input(img1)
    #Now it calls CNN model.predict for get feature values and flatten
    features = model.predict(process).flatten()
    feature_norm = features/norm(features)
    return feature_norm

def get_features(directory,model):
    filepath = []
    for file in os.listdir(directory):
        filepath.append(os.path.join(directory, file))

    feature_list = []

    for file in filepath:
        result = feature_extraction(file,model)
        feature_list.append(result)
        
    feature_list_array = np.array(feature_list)
    return feature_list_array,filepath

model = load_pretrained()
directory = 'data'
feature_list_array,filepath = get_features(directory,model)
pickle.dump(feature_list_array,open('embeddings.pkl','wb'))
pickle.dump(filepath,open('filenames.pkl','wb'))
#print(filepath)