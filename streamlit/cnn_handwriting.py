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
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas



import SessionState


# def main():
#     if 'button_id' not in st.session_state:
#         st.session_state['button_id'] = ''
#     if 'color_to_label' not in st.session_state:
#         st.session_state['color_to_label'] = {}
#     PAGES = {
#         "About": about,
#         "Basic example": full_app,
#         "Get center coords of circles": center_circle_app,
#         "Color-based image annotation": color_annotation_app,
#         "Download Base64 encoded PNG": png_export,
#         "Compute the length of drawn arcs": compute_arc_length,
#     }
#     page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
#     PAGES[page]()

#     with st.sidebar:
#         st.markdown("---")
#         st.markdown(
#             '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
#             unsafe_allow_html=True,
#         )
#         st.markdown(
#             '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
#             unsafe_allow_html=True,
#         )










# from svgpathtools import parse_path

# import SessionState

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

# Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     if st.button('save'):
#         st.write('Saved')
#         st.image(canvas_result.image_data)
#         print('DATAIMG',type(canvas_result.image_data))
#         # im = Image.fromarray(canvas_result.image_data) 
#         im = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
#         im.save("../streamlit/saved.png")


def png_export():
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("../img/img").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("../img/img").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    # if st.session_state["button_id"] == "":
    #     st.session_state["button_id"] = re.sub("\d+", "", str(uuid.uuid4()).replace("-", ""))

    # button_id = st.session_state["button_id"]
    file_path = "../img/img.png"

    # custom_css = f""" 
    #     <style>
    #         #{button_id} {{
    #             display: inline-flex;
    #             align-items: center;
    #             justify-content: center;
    #             background-color: rgb(255, 255, 255);
    #             color: rgb(38, 39, 48);
    #             padding: .25rem .75rem;
    #             position: relative;
    #             text-decoration: none;
    #             border-radius: 4px;
    #             border-width: 1px;
    #             border-style: solid;
    #             border-color: rgb(230, 234, 241);
    #             border-image: initial;
    #         }} 
    #         #{button_id}:hover {{
    #             border-color: rgb(246, 51, 102);
    #             color: rgb(246, 51, 102);
    #         }}
    #         #{button_id}:active {{
    #             box-shadow: none;
    #             background-color: rgb(246, 51, 102);
    #             color: white;
    #             }}
    #     </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()

        dl_link = (f'<a download="{file_path}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)


# png_export()

# data = st_canvas(update_streamlit=False, key="png_export")
# if data is not None and data.image_data is not None:
#     file_path = "../img/img"
#     img_data = data.image_data
#     im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
#     im.save(file_path, "PNG")

#     buffered = BytesIO()
#     im.save(buffered, format="PNG")
#     img_data = buffered.getvalue()
#     try:
#         # some strings <-> bytes conversions necessary here
#         b64 = base64.b64encode(img_data.encode()).decode()
#     except AttributeError:
#         b64 = base64.b64encode(img_data).decode()

#     dl_link = (
#         custom_css
#         + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
#     )
#     st.markdown(dl_link, unsafe_allow_html=True)


# data = st_canvas(update_streamlit=False, key="png_export")
file_path = "../img/img.png"

if canvas_result.image_data is not None:
    img_data = canvas_result.image_data
    im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
    im.save(file_path, "PNG")
    im.save('../img/img.png')

    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_data = buffered.getvalue()
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(img_data.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(img_data).decode()

    dl_link = (f'<a download="{file_path}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
    )
    st.markdown(dl_link, unsafe_allow_html=True)








image = Image.open('../img/fig/Conv2D_1.png')
st.image(
    image,
    caption='Conv2D_1',)

image = Image.open('../img/fig/Activation_1.png')
st.image(
    image,
    caption='Activation_1',)

image = Image.open('../img/fig/MaxPooling2D_1.png')
st.image(
    image,
    caption='MaxPooling2D_1',)

image = Image.open('../img/img.png')
st.image(
    image,
    caption='our test',)
    
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)

# print('NEWWW',type(canvas_result.image_data))