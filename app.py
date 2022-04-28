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
# Custom functions
from src.functions import *
from src.classifiers import *

######################################################
#                   Configuration
######################################################

st.set_page_config(page_title='Pattern recognition on Images/Videos with OpenCV', page_icon=':milky_way:', layout='wide')

#st.session_state
if 'something' not in st.session_state:
    st.session_state['something'] = False

st.title('Pattern recognition on Images/Videos with OpenCV')

st.markdown('Whatever.')


FRAME_WINDOW = st.image([], width=300, channels='BGR')
cam = cv2.VideoCapture(0)

webcam_capture = st.checkbox('Open Webcam')
while webcam_capture:
	# keep webcam connected
	_, frame = cam.read()
	# feature scan
	video_scan(frame, classifiers)
	# show video
	FRAME_WINDOW.image(frame)
else:
	cam.release()


# Load images
#images = {file:load_image(file) for file in file_urls.keys()}