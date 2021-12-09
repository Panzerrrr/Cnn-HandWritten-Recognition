import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from io import BytesIO
from pathlib import Path
import time

import base64
import os
import time

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap





     ################################################# PREDICT #########################################################


from tensorflow import keras
from tensorflow.keras.utils import *



import tensorflow as tf

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'my_model.h5')
model = keras.models.load_model(MODEL_DIR)
# st.write(MODEL_DIR)
def make_prediction():
    global new_img
    global img
    global pred

    img = canvas_result.image_data
    
    image_data = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image_data = image_data.resize((28, 28))
    image_data = image_data.convert('L')
    image_data = (tf.keras.utils.img_to_array(image_data)/255)
    image_data = image_data.reshape(1,28,28,1)
    new_img = tf.convert_to_tensor(image_data)

    pred = model.predict(new_img)
    print(pred)



def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))




######################################### APP ##############################################################



# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 25)
stroke_color = st.sidebar.color_picker("Stroke color hex: ","#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=500,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas",
)


# data = st_canvas(update_streamlit=False, key="png_export")
file_path = "../img/img.png"

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    button_clicked = st.button('prediction',on_click=make_prediction())


    if button_clicked:


        
        progress_bar = st.progress(1)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)


        viridis = plt.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        pink = np.array([248/256, 24/256, 148/256, 1])
        newcolors[:25, :] = pink
        newcmp = ListedColormap(newcolors)




        batch = new_img
        conv = model.layers[0]
        activation = conv(batch)
        # print(batch)
        print(conv)
        print(activation.shape)

        
        conv1 = model.layers[1]
        activation1 = conv1(activation)
        # print(batch)
        print(conv1)
        print(activation1.shape)


        conv3 = model.layers[2]
        activation2 = conv3(activation1)
        # print(batch)
        print(conv3)
        print(activation2.shape)


        conv3 = model.layers[3]
        activation3 = conv3(activation2)
        # print(batch)
        print(conv3)
        print(activation3.shape)

        conv4 = model.layers[4]
        activation4 = conv4(activation3)
        # print(batch)
        print(conv4)
        print(activation4.shape)

        conv5 = model.layers[5]
        activation5 = conv5(activation4)
        # print(batch)
        print(conv5)
        print(activation5.shape)

        n_filters =6    
        ix=1
        fig = plt.figure(figsize=(60,45))
        for i in range(30):
            # get the filters
            f = activation5[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,30,ix)
                plt.axis('off') 
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)


        # html_sdl = 

        st.text("\n")
        st.text("\n")     
        st.text("\n")
        st.text("\n")

        # st.markdown(html_sdl, unsafe_allow_html=True)

        
        n_filters =6
        ix=1
        fig = plt.figure(figsize=(50,35))
        for i in range(25):
            # get the filters
            f = activation5[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,25,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)


        st.text("\n")
        st.text("\n")



        n_filters =6
        ix=1
        fig = plt.figure(figsize=(40,25))
        for i in range(20):
            # get the filters
            f = activation4[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)


        
        st.text("\n")
        st.text("\n")



        n_filters =6
        ix=1
        fig = plt.figure(figsize=(30,15))
        for i in range(15):
            # get the filters
            f = activation3[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,15,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)

        
        st.text("\n")
        st.text("\n")

        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(10):
            # get the filters
            f = activation2[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,10,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)

        
        st.text("\n")
        st.text("\n")



        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(5):
            # get the filters
            f = activation1[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,5,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)

        
        st.text("\n")
        st.text("\n")



        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(2):
            # get the filters
            f = activation[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,2,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='viridis')
                ix+=1
        st.pyplot(fig)


        fig, ax = plt.subplots()
        plt.axis("off")
        ax.imshow(new_img[0], cmap=plt.get_cmap('viridis'))
        st.pyplot(fig)


        html_str = f"""
        <style>
        p.a {{
        font: bold 30px Courier;
        }}
        p.b {{
        font-family:sans-serif;
        color:Green;
        font-size: 42px;"
        }}
        
        </style>
        <p class="b">Prédictions du Modéle</p>
        <p class="a">{np.argmax(pred)}</p>

        <p class="b">Différente étapes de la prédiction </p>
        <p class="a">{np.argsort(pred)}</p>
        """

        st.markdown(html_str, unsafe_allow_html=True)

        st.balloons()



    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(new_img.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(new_img).decode()

    # dl_link = (f'<a download="{file_path}" onclick="{make_prediction()}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
    # )

    
    # st.markdown(dl_link, unsafe_allow_html=True)



    ############################################## PLOT ###############################################################