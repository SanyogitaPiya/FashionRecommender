# Fashion Recommender System
## Area of Choice: Fashion
- Due to increased usage of online purchasing fashion recommender systems as gained high popularity
- These systems use algorithms to provide recommendations to customers based on their preferences and previous purchase history.
- Our project is targeted at users who are looking for online purchases for the first time.
- By using an input image instead of the user’s previous purchases and history, our model can provide more accurate recommendations that reflect the user’s current preferences.
## Method:
- Fashion Recommendation system that generates recommendations based on an input image
- Users can upload certain fashion and styles related images such as jeans, jewelry and shirts 
- The system will give similar images that is in the dataset
## Dataset:
- The dataset for this project is going to be the images of fashion product downloaded from Kaggle.
- The total size of the images is 15GB with over 44,000 high-resolution images.
## CNN and KNN:
- A Convolutional Neural Network (CNN) is a type of deep learning algorithm that is commonly used for image recognition and classification tasks.
- It consists of multiple layers of small, interconnected processing units, called neurons, which are designed to detect specific features in images.
- In our project, features of these images are extracted to learn the contents using deep learning model.
- We then use the Nearest neighbour algorithm to find the most relevant products based on the input image and generate recommendations.
## Experimental Steps:
- Used ResNet-50 as a CNN architecture
- ResNet-50: pre-trained on 1.2 million ImageNet dataset, so it can be used for both feature extraction from images and image classification
- Used to extract features from both the dataset and the input image and return the normalized feature vector
- Took around 8 hours for the dataset feature extraction
- Saved in a pickle file so that it can be reused later without again having to re-extract the features everytime the system is used.
- After feature extraction is completed from the input image, we utilized NearestNeighbors class from scikit-learn to find the 5 nearest images to the one from the input image
- We used Streamlit for the user interface and to display the 5 similar images to the user






