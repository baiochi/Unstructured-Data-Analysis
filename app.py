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
from src.functions   import *
from src.classifiers import *
from src.image_db    import *
from src.st_render   import *

######################################################
#                 Page Configuration
######################################################

st.set_page_config(page_title='Pattern recognition on Images/Videos with OpenCV', page_icon=':milky_way:', layout='wide')

#st.session_state
if 'something' not in st.session_state:
    st.session_state['something'] = False
if 'active_classifiers' not in st.session_state:
    st.session_state['active_classifiers'] = {}
st.session_state.classifiers = classifiers
# st.session_state

st.title('Pattern recognition on Images/Videos with OpenCV')


######################################################
#                 Sidebar Settings
######################################################

st.sidebar.header('Settings')

# Render form to add new classifiers via file upload
#render_add_new_classifier(classifiers)

# Select active classifiers
with st.sidebar.expander(label='Select classifiers', expanded=False):

	# User form to select classifiers
	with st.form(key='select_classifiers_form', clear_on_submit=True):
		_active_classifiers = {}
		for key_index, classifier_name in enumerate(st.session_state.classifiers):
			_current_classifier = st.session_state.classifiers[classifier_name]
			_current_checkbox   = st.checkbox(label=f'{classifier_name}', key=key_index)
			if _current_checkbox and ( classifier_name not in st.session_state.active_classifiers.keys() ):
				_active_classifiers[classifier_name] = _current_classifier

		submit_active_classifiers = st.form_submit_button('Apply')

		# Store new classifier when submit button is hit.
		if submit_active_classifiers:
			st.session_state.active_classifiers = _active_classifiers

st.sidebar.markdown(f"Active classifiers: {list(st.session_state['active_classifiers'].keys())}")

# Change classifier param
with st.sidebar.expander(label='Modify params', expanded=False):

	_c_selected = st.selectbox(label='Select classifier', options=st.session_state.classifiers.keys())
	if _c_selected:
		c_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, 
										value=int(st.session_state.classifiers[_c_selected]['minNeighbors']))
		c_color        = st.color_picker('Select color', 
										value=rgb_to_hex(st.session_state.classifiers[_c_selected]['color'])); 
		c_color = hex_to_rgb(c_color)

		# Build classifier structure
		updated_classifier = {
			'minNeighbors': c_minNeighbors,
			'color'       : c_color
		}

		st.session_state.classifiers[_c_selected].update(updated_classifier)
		

######################################################
#                  Image Settings
######################################################

st.sidebar.header('Image Settings')

with st.sidebar.expander(label='Options', expanded=False):

	image_choice = st.radio('Select an option to load a image:', options=('Upload file', 'Type image url', 'Select from database'))

	if image_choice == 'Upload file':
		upload_image = st.file_uploader(label='Upload an image.', type='xml')
	if image_choice == 'Type image url':
		image_url = st.text_input('Insert the url of an image')
	if image_choice == 'Select from database':
		selected_image = st.selectbox('Select an image:', options=image_db)


	
	
######################################################
#                  Video Settings
######################################################

st.sidebar.header('Video Settings')
with st.sidebar.expander(label='Options', expanded=False):
	st.text('Building...')

######################################################
#                  Main Page
######################################################

st.subheader('Select between image or video recognition')

webcam_capture = st.checkbox('Open Webcam')
cam = cv2.VideoCapture(0)
window_frame = st.image([], width=200)

while webcam_capture:
	# keep webcam connected
	_, frame = cam.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# feature scan
	#video_scan(frame, st.session_state.classifiers)
	# show video
	window_frame.image(frame)
else:
	cam.release()


# Load images
#images = {file:load_image(file) for file in file_urls.keys()}