import streamlit as st
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import webbrowser
import pandas as pd
from keras.models import Model
# from tensorflow.keras.utils import load_img, img_to_array
# from tensorflow.keras.models import Sequential, model_from_json
import os
from io import StringIO, BytesIO
from keras.applications.imagenet_utils import preprocess_input
import cv2
from keras.models import load_model, model_from_json

# resnet50


df = pd.read_csv('styles.csv', error_bad_lines=False)


# unique_types = ['Jeans', 'Shirts', 'Watches', 'Track Pants', 'Tshirts', 'Socks', 'Casual Shoes', 'Belts', 'Flip Flops', 'Handbags', 'Tops', 'Bra', 'Sandals', 'Shoe Accessories', 'Sweatshirts', 'Deodorant', 'Formal Shoes', 'Bracelet', 'Lipstick', 'Flats', 'Kurtas', 'Waistcoat', 'Sports Shoes', 'Shorts', 'Briefs', 'Sarees', 'Perfume and Body Mist', 'Heels', 'Sunglasses', 'Innerwear Vests', 'Pendant', 'Nail Polish', 'Laptop Bag', 'Scarves', 'Rain Jacket', 'Dresses', 'Night suits', 'Skirts', 'Wallets', 'Blazers', 'Ring', 'Kurta Sets', 'Clutches', 'Shrug', 'Backpacks', 'Caps', 'Trousers', 'Earrings', 'Camisoles', 'Boxers', 'Jewellery Set', 'Dupatta', 'Capris', 'Lip Gloss', 'Bath Robe', 'Mufflers', 'Tunics', 'Jackets', 'Trunk', 'Lounge Pants', 'Face Wash and Cleanser', 'Necklace and Chains', 'Duffel Bag', 'Sports Sandals', 'Foundation and Primer',
#                 'Sweaters', 'Free Gifts', 'Trolley Bag', 'Tracksuits', 'Swimwear', 'Shoe Laces', 'Fragrance Gift Set', 'Bangle', 'Nightdress', 'Ties', 'Baby Dolls', 'Leggings', 'Highlighter and Blush', 'Travel Accessory', 'Kurtis', 'Mobile Pouch', 'Messenger Bag', 'Lip Care', 'Face Moisturisers', 'Compact', 'Eye Cream', 'Accessory Gift Set', 'Beauty Accessory', 'Jumpsuit', 'Kajal and Eyeliner', 'Water Bottle', 'Suspenders', 'Lip Liner', 'Robe', 'Salwar and Dupatta', 'Patiala', 'Stockings', 'Eyeshadow', 'Headband', 'Tights', 'Nail Essentials', 'Churidar', 'Lounge Tshirts', 'Face Scrub and Exfoliator', 'Lounge Shorts', 'Gloves', 'Mask and Peel', 'Wristbands', 'Tablet Sleeve', 'Ties and Cufflinks', 'Footballs', 'Stoles', 'Shapewear', 'Nehru Jackets', 'Salwar', 'Cufflinks', 'Jeggings', 'Hair Colour', 'Concealer', 'Rompers', 'Body Lotion']
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
st.set_page_config(page_title="Image Recommendation System", layout="wide")

# @st.cache(allow_output_mutation=True)
# model = load_model("final_model.h5")
json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("final_model.h5")

col1, mid, col2 = st.columns([1, 15, 100])
with col1:
    st.image('./Fashion_Camera2.jpg', width=150)
with col2:
    #st.write('A Name')
    st.markdown('<h1 style="color: red;font-size: 70px;">FashCam</h1>',
                unsafe_allow_html=True)
    st.markdown('<h1 style="color: red;font-size: 30px;">...an Image Search Engine</h1>',
                unsafe_allow_html=True)

st.markdown("Our idea is to build a new search engine **:red[_FashCam_]**:camera:.We have developed a cutting-edge image recognition technology that makes it easy to find the fashion you want. With **:red[_FashCam_]**:camera:, you can simply take a picture of an item or upload an image and our algorithm will match it with similar products available for purchase online. It's that simple!")
st.sidebar.write("## Upload or Take a Picture")

# Upload the image file
file = st.sidebar.file_uploader(
    "Choose an image from your computer", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    #size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    img_reshape = img_reshape[..., np.newaxis]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.sidebar.subheader(
        "Please upload a product image using the browse button :point_up:")
    st.sidebar.write(
        "Sample image can be found [here](https://github.com/prachiagrl83/WBS/tree/Prachi/Sample_images)!")

else:
    st.sidebar.subheader(
        "Thank you for uploading the image. Below you can see image which you have just uploaded!")
    st.subheader("Scroll down to see the Top Similar Products...")
    st.sidebar.image(file, width=250)
    image = Image.open(file)
    predictions = import_and_predict(image, model)
    result = st.write("we are searching in our shop for similar " +
                      unique_types[np.argmax(predictions)])
