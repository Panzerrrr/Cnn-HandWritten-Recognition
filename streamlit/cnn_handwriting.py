import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from io import BytesIO
from pathlib import Path
import os
import time

import base64
import os
import time

import pandas as pd

import matplotlib.pyplot as plt






     ################################################# PREDICT #########################################################



from keras.models import load_model

# load model after export

import pickle   


# predictions = model.predict([X_test])
# print(predictions)

# pip install opencv_python
# import cv2
# img = cv2.imread('../img/img.png')
# plt.imshow(img)

# # print(img.shape)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # print(gray.shape)

# resize = cv2.resize(gray, (28,28),interpolation=cv2.INTER_AREA)
# resize.shape
# print(plt.imshow(resize))


import tensorflow as tf

# IMG_SIZE = 28

# new_img = tf.keras.utils.normalize(resize,axis=1)keras
# new_img = np.array(new_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# print(new_img.shape)


# pred = model.predict(new_img)
# print(pred)


# print(np.argsort(pred))
# MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'my_model.h5')
# model = load_model('C:\Users\Simplon\projets\Cnn-HandWritten-Recognition\streamlit\my_model.h5')
# model = load_model(MODEL_DIR)
# model = load_model('my_model.h5')

# MODEL_DIR = os.path.join(os.path.dirname(__file__), 'my_model.h5')
# if not os.path.isdir(MODEL_DIR):
#     os.system(r'my_model.h5')
# model = load_model(r'my_model.h5')


# MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'my_model.h5')
# if not os.path.isdir(MODEL_DIR):
#     os.system('runipy train.ipynb')

model = os.path.join(os.path.dirname('__file__'), 'my_model.h5')
model = load_model(model)




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


def make_prediction():
    global new_img
    global img
    # predictions = model.predict([X_test]) 
    # img = cv2.imread('../img/img.png')   
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # resize = cv2.resize(gray, (28,28),interpolation=cv2.INTER_AREA)
    # IMG_SIZE = 28

    # new_img = tf.keras.utils.normalize(resize,axis=1)
    # new_img = np.array(new_img).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    # print(new_img.shape)
    # pred = model.predict(new_img)
    # print(pred)
    # print(np.argsort(pred))


    img = canvas_result.image_data
    
    image_data = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image_data = image_data.resize((28, 28))
    image_data = image_data.convert('L')
    image_data = (tf.keras.utils.img_to_array(image_data)/255)
    image_data = image_data.reshape(1,28,28,1)
    new_img = tf.convert_to_tensor(image_data)
    

    # print(new_img.shape)
    pred = model.predict(new_img)
    print(pred)
    print(np.argsort(pred))

        # return new_img

# def png_export():
#     st.markdown(
#         """
#     Realtime update is disabled for this demo. 
#     Press the 'Download' button at the bottom of canvas to update exported image.
#     """
#     )
#     try:
#         Path("../img/img").mkdir()
#     except FileExistsError:
#         pass

#     # Regular deletion of tmp files
#     # Hopefully callback makes this better
#     now = time.time()
#     N_HOURS_BEFORE_DELETION = 1
#     for f in Path("../img/img").glob("*.png"):
#         st.write(f, os.stat(f).st_mtime, now)
#         if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
#             Path.unlink(f)

#     # button_id = st.session_state["button_id"]
#     file_path = "../img/img.png"

#     # custom_css = f""" 
#     #     <style>
#     #         #{button_id} {{
#     #             display: inline-flex;
#     #             align-items: center;
#     #             justify-content: center;
#     #             background-color: rgb(255, 255, 255);
#     #             color: rgb(38, 39, 48);
#     #             padding: .25rem .75rem;
#     #             position: relative;
#     #             text-decoration: none;
#     #             border-radius: 4px;
#     #             border-width: 1px;
#     #             border-style: solid;
#     #             border-color: rgb(230, 234, 241);
#     #             border-image: initial;
#     #         }} 
#     #         #{button_id}:hover {{
#     #             border-color: rgb(246, 51, 102);
#     #             color: rgb(246, 51, 102);
#     #         }}
#     #         #{button_id}:active {{
#     #             box-shadow: none;
#     #             background-color: rgb(246, 51, 102);
#     #             color: white;
#     #             }}
#     #     </style> """

#     # data = st_canvas(update_streamlit=False, key="png_export")
#     # if data is not None and data.image_data is not None:
#     #     img_data = data.image_data
#     #     im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
#     #     im.save(file_path, "PNG")

#     #     buffered = BytesIO()
#     #     im.save(buffered, format="PNG")
#     #     img_data = buffered.getvalue()
#     #     try:
#     #         # some strings <-> bytes conversions necessary here
#     #         b64 = base64.b64encode(img_data.encode()).decode()
#     #     except AttributeError:
#     #         b64 = base64.b64encode(img_data).decode()

#     #     dl_link = (f'<a download="{file_path}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
#     #     )
#     #     st.markdown(dl_link, unsafe_allow_html=True)


# data = st_canvas(update_streamlit=False, key="png_export")
file_path = "../img/img.png"

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    button_clicked = st.button('prediction',on_click=make_prediction())

    # img_data = canvas_result.image_data
    # im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
    # im.save(file_path, "PNG")
    # im.save('../img/img.png')

    # buffered = BytesIO()
    # im.save(buffered, format="PNG")
    # img_data = buffered.getvalue()


   
    
    # image_data = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    # image_data = image_data.resize((28, 28))
    # image_data = image_data.convert('L')
    # image_data = (tf.keras.utils.img_to_array(image_data)/255)
    # image_data = image_data.reshape(1,28,28,1)
    # test_x = tf.convert_to_tensor(image_data)
    
    
    #Version CV2 non supporté par streamlit
    #img2 = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    #img = img2
    #img = img / 255.0
    #img = img.reshape(-1,28,28,1)
    #rescaled = cv2.resize(img2, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    #st.write('Model Input')
    # st.image(image_data)

    


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
        plt.savefig("../img/fig/{model.layers.name[0]}.png")
        # # plot the fig
        # plt.show()

        st.write('predction en cours')
        image = Image.open('../img/fig/{model.layers.name[0]}.png')
        st.image(
            image,
            caption='model.layers.name[0]',)


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
        plt.savefig("../img/fig/{model.layers.name[1]}.png")
        # # plot the fig
        # plt.show()



        image = Image.open('../img/fig/{model.layers.name[1]}.png')
        st.image(
            image,
            caption='model.layers.name[1]',)


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
        plt.savefig("../img/fig/{model.layers.name[2]}.png")
        # # plot the fig
        # plt.show()

        image = Image.open('../img/fig/{model.layers.name[2]}.png')
        st.image(
            image,
            caption='model.layers.name[2]',)


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
        plt.savefig("../img/fig/{model.layers.name[3]}.png")
        # # plot the fig
        # plt.show()


        image = Image.open('../img/fig/{model.layers.name[3]}.png')
        st.image(
            image,
            caption='model.layers.name[3]',)


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
        plt.savefig("../img/fig/{model.layers.name[4]}.png")
        # # plot the fig
        # plt.show()


        image = Image.open('../img/fig/{model.layers.name[4]}.png')
        st.image(
            image,
            caption='model.layers.name[4]',)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(new_img.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(new_img).decode()

    # dl_link = (f'<a download="{file_path}" onclick="{make_prediction()}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
    # )

    
    # st.markdown(dl_link, unsafe_allow_html=True)



    ############################################## PLOT ###############################################################



# def plot_nodes():

#     for nodes in range(3):
#         batch = new_img
#         conv = model.layers[nodes]
#         activation = conv(batch)
#         # print(batch)
#         print(conv)
#         print(activation.shape)

#         import matplotlib.pyplot as plt

#         n_filters =6
#         ix=1
#         fig = plt.figure(figsize=(20,15))
#         for i in range(20):
#             # get the filters
#             f = activation[:,:,:,i]
#             for j in range(1):
#                 # subplot for 6 filters and 3 channels
#                 plt.subplot(1,20,ix)
#                 plt.axis('off')
#                 plt.imshow(f[j,:,:] ,cmap='gray')
#                 ix+=1
#         # save the fig
#         plt.savefig("../img/fig/{model.layers.name}.png")
#         # # plot the fig
#         # plt.show()

















    
