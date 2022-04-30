######################################################
#                Libraries and APIs
######################################################

# Supress warnings
import warnings; warnings.filterwarnings('ignore')
# Array manipulation
import numpy as np
# Image processing and visualization
import cv2
import matplotlib.pyplot as plt
from PIL.Image import open as open_image
# HTTP requests
from urllib.request import urlopen
# Web rendering API
import streamlit as st
import mediapipe as mp
# Custom functions
from src.functions   import *
from src.defines     import *
from src.processors  import *
from src.image_db    import *
from src.st_render   import *

######################################################
#                 Page Configuration
######################################################

st.set_page_config(page_title='Pattern recognition on Images/Videos with OpenCV', page_icon=':milky_way:', layout='wide')

# if 'foo' not in st.session_state:
#     st.session_state['foo'] = False
if 'image' not in st.session_state:
    st.session_state['image'] = False
if 'classifiers' not in st.session_state:
	st.session_state['classifiers'] = load_classifiers()
# st.session_state

st.title('Pattern recognition on Images/Videos with OpenCV')

######################################################
#                 Sidebar Settings
######################################################

image_detection_page      = 'Image Detection'
video_detection_page      = 'Video Detection'
live_video_detection_page = 'Live Video Detection'

app_mode = st.sidebar.selectbox(label='Choose the app mode',
			options=[
				image_detection_page,
				video_detection_page,
				live_video_detection_page
			])

st.subheader(app_mode)

if   app_mode == image_detection_page:
	show_image_detection_page()
elif app_mode == video_detection_page:
	show_video_detection_page()
elif app_mode == live_video_detection_page:
	show_live_video_detection_page()

# Sidebar Params configuration
adjust_classifier_params()