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

    img = canvas_result.image_data
    
    image_data = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image_data = image_data.resize((28, 28))
    image_data = image_data.convert('L')
    image_data = (tf.keras.utils.img_to_array(image_data)/255)
    image_data = image_data.reshape(1,28,28,1)
    new_img = tf.convert_to_tensor(image_data)

    pred = model.predict(new_img)
    print(pred)
    print(np.argsort(pred))


def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


# image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
#     if image_file is not None:
#         file_details = {"FileName":image_file.name,"FileType":image_file.type}
#         st.write(file_details)
#         img = load_image(image_file)
#         st.image(img,height=250,width=250)
#         with open(os.path.join("tempDir",image_file.name),"wb") as f: 
#         f.write(image_file.getbuffer())         
#         st.success("Saved File")



        




######################################### APP ##############################################################



# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
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




def load_image(image_file):
    img = Image.open(image_file)
    return img

# data = st_canvas(update_streamlit=False, key="png_export")
file_path = "../img/img.png"

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    button_clicked = st.button('prediction',on_click=make_prediction())





    # # To See details
    # file_details = {"filename":canvas_result, "filetype":canvas_result,
    #                 "filesize":canvas_result}
    # st.write(file_details)

    # # To View Uploaded Image
    # st.image(load_image(canvas_result.image_data),width=250)



    if button_clicked:

        batch = new_img
        conv = model.layers[0]
        activation = conv(batch)
        # print(batch)
        print(conv)
        print(activation.shape)


        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(20):
            # get the filters
            f = activation[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='gray')
                ix+=1
        # save the fig
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod1.png'))
        # # plot the fig
        # plt.show()

        # img_data = data.image_data
        # im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        # im.save(file_path, "PNG")

        # buffered = BytesIO()
        # im.save(buffered, format="PNG")
        # img_data = buffered.getvalue()

        st.write('predction en cours')
        image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod1.png'))
        st.image(
            image,
            caption='mod1',)


        conv1 = model.layers[1]
        activation1 = conv1(activation)
        # print(batch)
        print(conv1)
        print(activation1.shape)

        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(20):
            # get the filters
            f = activation1[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='gray')
                ix+=1
        # save the fig
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod2.png'))
        # # plot the fig
        # plt.show()



        image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod2.png'))
        st.image(
            image,
            caption='mod2',)


        conv3 = model.layers[2]
        activation2 = conv3(activation1)
        # print(batch)
        print(conv3)
        print(activation2.shape)

        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(20):
            # get the filters
            f = activation2[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='gray')
                ix+=1
        # save the fig
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod3.png'))
        # # plot the fig
        # plt.show()

        image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod3.png'))
        st.image(
            image,
            caption='mod3',)


        conv3 = model.layers[3]
        activation3 = conv3(activation2)
        # print(batch)
        print(conv3)
        print(activation3.shape)


        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(20):
            # get the filters
            f = activation3[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='gray')
                ix+=1
        # save the fig
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod4.png'))
        # # plot the fig
        # plt.show()


        image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod4.png'))
        st.image(
            image,
            caption='mod4',)


        conv4 = model.layers[4]
        activation4 = conv4(activation3)
        # print(batch)
        print(conv4)
        print(activation4.shape)


        n_filters =6
        ix=1
        fig = plt.figure(figsize=(20,15))
        for i in range(20):
            # get the filters
            f = activation4[:,:,:,i]
            for j in range(1):
                # subplot for 6 filters and 3 channels
                plt.subplot(1,20,ix)
                plt.axis('off')
                plt.imshow(f[j,:,:] ,cmap='gray')
                ix+=1
        # save the fig
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod5.png'))
        # # plot the fig
        # plt.show()


        image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mod5.png'))
        st.image(
            image,
            caption='mod5',)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(new_img.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(new_img).decode()

    # dl_link = (f'<a download="{file_path}" onclick="{make_prediction()}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
    # )

    
    # st.markdown(dl_link, unsafe_allow_html=True)



    ############################################## PLOT ###############################################################