import numpy as np
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st
import pandas as pd
from PIL import Image
import os
import pickle


def userInterface():
    st.set_page_config(page_title="Fashion Recommender System")
    st.write("""
    # Fashion Recommender System

    Welcome to our fashion recommender system! This system uses machine learning algorithms CNN and KNN to recommend clothing items based on your uploaded photo.

    """)
    st.write("""
    ---
    Machine Learning Project by:
    - Sanyogita Piya
    - Nirdesh Sakh
    """)
    st.write("""
    ---
    Please upload a photo here...
    """)


def load_pretrained():
    # Load pre-trained ResNet50 model
    pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add GlobalAveragePooling2D layer to the ResNet50 model for fixed-length vector that can be fed into the subsequent layers
    x = pretrained.output
    x = GlobalAveragePooling2D()(x)

    # Create a new model with the ResNet50 base model and the GlobalAveragePooling2D layer
    model = tensorflow.keras.Model(inputs=pretrained.input, outputs=x)
    return model


def feature_extraction(filename, model):
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    img1 = np.expand_dims(img, axis=0)
    process = preprocess_input(img1)
    features = model.predict(process).flatten()
    feature_norm = features / norm(features)
    return feature_norm


def predict(query_feature, feature_list_array):
    k = 5  # 5 nearest neighbors
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn_model.fit(feature_list_array)
    distances, indices = knn_model.kneighbors([query_feature])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        #print(os.path.join('uploads',uploaded_file.name))
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
def get_uploaded_file(feature_list_array,labels):
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            # display the file
            display_image = Image.open(uploaded_file)
            max_size = (500, 500)  
            display_image.thumbnail(max_size, Image.ANTIALIAS)
            st.image(display_image)
            # feature extract
            features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
            indices = predict(features,feature_list_array)
            col1,col2,col3,col4,col5 = st.columns(5)
            with col1:
                st.image(labels[indices[0][0]])
            with col2:
                st.image(labels[indices[0][1]])
            with col3:
                st.image(labels[indices[0][2]])
            with col4:
                st.image(labels[indices[0][3]])
            with col5:
                st.image(labels[indices[0][4]])
        else:
            st.header("Some error occured in file upload")


model = load_pretrained()
userInterface()
feature_list_array = np.array(pickle.load(open('embeddings.pkl','rb')))
labels = pickle.load(open('filenames.pkl','rb'))
# feature_list_array = np.load('features.npy')
# labels = np.load('labels.npy')
print(feature_list_array)
get_uploaded_file(feature_list_array,labels) 



