
# TensorFlow and tf.keras


from keras.applications import ResNet50
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, model_from_json

from keras.models import Model
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from keras.models import Model
# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.models import Sequential, model_from_json
import os
from keras.applications.imagenet_utils import preprocess_input
import cv2
from keras.models import load_model, model_from_json
# import resnet50
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# get the array with the class names from the folder
# read foldernames
import os
import glob
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
unique_types = ['Backpacks',
                'Belts',
                'Bra',
                'Capris',
                'Caps-hats',
                'Casual Shoes',
                'Clutches',
                'Dresses',
                'Earrings',
                'Flip Flops',
                'Handbags',
                'Heels',
                'Jeans',
                'Jewellery_Set',
                'Kurtas',
                'Leggings',
                'Outwear',
                'pijamas',
                'Ring',
                'Salwar',
                'Sandals',
                'Scarves',
                'Shirts',
                'Shorts',
                'Skirts',
                'Socks',
                'Sports Shoes',
                'Sunglasses',
                'Sweatshirts',
                'Swimwear',
                'Tops',
                'Track Pants',
                'Tracksuits',
                'Trousers',
                'Tshirts',
                'Tunics',
                'Wallets',
                'Watches']


def get_prediction_resnet(img_path):

    # Load an image to use for prediction
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the predictions from the model
    features = model.predict(x)

    # Print the top 5 predictions
    predictions = decode_predictions(features, top=2)
    # for p in predictions[0]:
    #     print(f"Class: {p[1]}, Probability: {p[2]:.2f}")
    # feature extractor with resnet50
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    # get the features of the image
    img_features = feat_extractor.predict(x)
    return predictions[0][0], img_features


# function that gets a list of images and returns the features of those images
def get_features(img_paths, category_folder_name):
    importedImages = []
    for f in img_paths:
        filename = f
        original = load_img(filename, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)

        importedImages.append(image_batch)
    images = np.vstack(importedImages)
    processed_imgs = preprocess_input(images.copy())
    # load the model

    # Load the pre-trained ResNet50 model, with the top layer removed
    model = ResNet50(weights='imagenet', include_top=True)
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    imgs_features = feat_extractor.predict(processed_imgs)
    print("features successfully extracted!")
    # save it as csv file
    np.savetxt(f'{category_folder_name}.csv', imgs_features, delimiter=",")
    return imgs_features


# get the image names
# make the path for each image
shopfiles = ['finalDataset/Outwear/' +
             f for f in os.listdir('finalDataset/Outwear')]
# get_features(shopfiles, "0utwear")


def get_prediction_tuning(img_path):
    # load json and create model
    json_file = open('final_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("final_model.h5")
    # Load an image to use for prediction
    img = load_img(img_path, target_size=(224, 224, 3))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy',
                         metrics=['accuracy'])
    result = loaded_model.predict(x)

    return (result, unique_types[np.argmax(result)])


# function that takes an image and available_class as input and applies the cosine similarity with the features from csv to get the closest images
def get_closest_images(image, available_class):
    nb_closest_images = 5

    # get an NOT a path but an actual image as input and repeat the same process
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = image[np.newaxis, ...]
    img_reshape = img_reshape[..., np.newaxis]
    # load the features from csv
    features = np.genfromtxt(f'{available_class}.csv', delimiter=',')
    # get the features of the image
    model = ResNet50(weights='imagenet', include_top=True)
    feat_extractor = Model(
        inputs=model.input, outputs=model.get_layer("avg_pool").output)
    feat_extractor.summary()
    img_features = feat_extractor.predict(img_reshape)
    # get the cosine similarity
    cosSimilarities = cosine_similarity(img_features, features)
    closest_imgs_indexes = cosSimilarities.argsort()[0][-nb_closest_images:]
    # similarity score of the closest images
    closest_imgs_similarities = cosSimilarities[0][closest_imgs_indexes]

    # get the closest images
    closest_imgs = [shopfiles[i]
                    for i in closest_imgs_indexes if i < len(shopfiles)]
    similar = []
    for i in range(0, nb_closest_images):

        #
        # selecting the most similar with threshold 0.3
        if closest_imgs_similarities[i] > 0.3:
            # give the path of the image
            similar.append(closest_imgs[i])

    return similar


# print(get_closest_images(
#     r"C:\Users\mrpal\OneDrive\Desktop\fashcam-final\finalDataset\Belts\3721.jpg", "Outwear"))
