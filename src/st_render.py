
######################################################
#             Streamlit Rendering Functions
######################################################

import urllib
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from src.functions   import *
from src.defines     import IMAGES_URL
from src.processors  import VideoProcessor

RTC_CONFIGURATION = RTCConfiguration(
    {
      "RTCIceServer": [{
        "urls": ["turn:turn.xxx.dev:5349"],
        "username": "user",
        "credential": "password",
      }]
    }
)

# Modifty classifiers params
def adjust_classifier_params():

	# Create sidebar
	with st.sidebar.expander('Classifier Parameters'):

		# Select param
		_c_selected = st.selectbox(label='Select classifier to modify', options=st.session_state['classifiers'].keys())

		if _c_selected:
			# change minNeighbors
			c_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, 
											value=int(st.session_state.classifiers[_c_selected]['minNeighbors']))
			# change frame color - value is obtained in rgb -> hex and -> rgb again
			c_color        = st.color_picker('Select color', 
											value=rgb_to_hex(st.session_state.classifiers[_c_selected]['color'])); 
			c_color = hex_to_rgb(c_color)

			# Build classifier structure
			updated_classifier = {
				'minNeighbors': c_minNeighbors,
				'color'       : c_color
			}
			# Update classifer
			st.session_state['classifiers'][_c_selected].update(updated_classifier)
		
		# Reset to default values
		with st.form('reset_classifiers'):
			st.write('Reset params to default values')
			submitted = st.form_submit_button('Submit')
			if submitted:
				st.session_state['classifiers'] = load_classifiers()
				st.success('Values set to default')


def image_classifier_selection(frame):
	
	classifier_list = st.multiselect('Active classifiers', options=st.session_state['classifiers'])
	active_classifiers = {i:st.session_state['classifiers'][i] for i in classifier_list }
	frame_scan(frame, active_classifiers)

def show_image_detection_page():

	# SIDEBAR
	with st.sidebar.expander(label='Image Options', expanded=False):

		image_opt_choice = st.radio('Select an option to load a image:', options=('Upload file', 'Type image url', 'Select from database'))

		if image_opt_choice == 'Upload file':
			upload_image = st.file_uploader(label='Upload an image.', type='xml')
			if upload_image:
				st.session_state['image'] = load_image_file(upload_image)
		if image_opt_choice == 'Type image url':
			image_url = st.text_input('Insert the url of an image')
			if image_url:
				st.session_state['image'] = load_image_url(image_url)
		if image_opt_choice == 'Select from database':
			sample_image = st.selectbox('Select an image:', options=IMAGES_URL.keys())
			if sample_image:
				st.session_state['image'] = load_image_url(IMAGES_URL[sample_image])

	# MAIN
	if isinstance(st.session_state['image'], np.ndarray):

		_original_image = st.session_state['image'].copy()

		# Select patterns to detect
		image_classifier_selection(st.session_state['image'])
		# Print image
		st.image(st.session_state['image'], width=600)

		# Reset image without drawings
		st.session_state['image'] = _original_image

		
			
def show_video_detection_page():
	pass

def show_live_video_detection_page():

	ctx = webrtc_streamer(key='main_video',
						rtc_configuration=RTC_CONFIGURATION,
						video_processor_factory=VideoProcessor)
	if ctx.video_processor:
		ctx.video_processor.classifier_list = st.multiselect('Active classifiers', options=st.session_state['classifiers'])



######################################################
#                 Deprecated
######################################################

# Render form to add new classifiers via file upload
def render_add_new_classifier(classifiers):
	with st.sidebar.expander(label='Add a new classifier', expanded=False):

		# User form to upload file
		with st.form(key='upload_form'):

			# Section to upload file and define parameters
			uploaded_classifier = st.file_uploader(label='Upload a xml file.', type='xml')
			#classifier_url = st.text_input(label='Classifier name')
			classifier_name = st.text_input(label='Classifier name')
			up_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, value=5)
			up_color        = st.color_picker('Select color', value='#FF0000'); up_color = hex_to_rgb(up_color)
			up_is_sub       = st.checkbox('Classifier is a sub feature')
			up_sub_class    = st.multiselect(label='Select sub classifiers (if any)', default=None, options=[key for key in classifiers.keys() if classifiers[key]['is_sub']])
			if up_sub_class:
				up_sub_search = True
			else:
				up_sub_search = False

			# Build classifier structure
			new_classifier = {
				#'classifier'  : cv2.CascadeClassifier(uploaded_classifier),
				'minNeighbors': up_minNeighbors,
				'color'       : up_color,
				'sub_search'  : up_sub_search,
				'is_sub'      : up_is_sub,
				'sub_class'   : up_sub_class
			}

			submit_upload = st.form_submit_button('Submit')

			# Store new classifier when submit button is hit.
			if submit_upload:
				if not uploaded_classifier:
					st.error('Please upload a file.')
				elif not classifier_name:
					st.error('Please insert a name for the classifier.')
				else:
					st.session_state.classifiers[classifier_name] = new_classifier
					st.success('File added to classifiers list.')

######################################################
#                  Image Settings
######################################################


# Select active classifiers
# with st.sidebar.expander(label='Select classifiers', expanded=False):

# 	# User form to select classifiers
# 	with st.form(key='select_classifiers_form', clear_on_submit=True):
# 		_active_classifiers = {}
# 		for key_index, classifier_name in enumerate(st.session_state.classifiers):
# 			_current_classifier = st.session_state.classifiers[classifier_name]
# 			_current_checkbox   = st.checkbox(label=f'{classifier_name}', key=key_index)
# 			if _current_checkbox and ( classifier_name not in st.session_state.active_classifiers.keys() ):
# 				_active_classifiers[classifier_name] = _current_classifier

# 		submit_active_classifiers = st.form_submit_button('Apply')

# 		# Store new classifier when submit button is hit.
# 		if submit_active_classifiers:
# 			st.session_state.active_classifiers = _active_classifiers

# st.sidebar.markdown(f"Active classifiers: {list(st.session_state['active_classifiers'].keys())}")

# # Change classifier param
# with st.sidebar.expander(label='Modify params', expanded=False):

# 	_c_selected = st.selectbox(label='Select classifier', options=st.session_state.classifiers.keys())
# 	if _c_selected:
# 		c_minNeighbors = st.number_input(label='minNeighbors', min_value=1, max_value=100, 
# 										value=int(st.session_state.classifiers[_c_selected]['minNeighbors']))
# 		c_color        = st.color_picker('Select color', 
# 										value=rgb_to_hex(st.session_state.classifiers[_c_selected]['color'])); 
# 		c_color = hex_to_rgb(c_color)

# 		# Build classifier structure
# 		updated_classifier = {
# 			'minNeighbors': c_minNeighbors,
# 			'color'       : c_color
# 		}

# 		st.session_state.classifiers[_c_selected].update(updated_classifier)


######################################################
#                  Video Settings
######################################################


# webcam_capture = st.checkbox('Open Webcam')
# cam = cv2.VideoCapture(0)
# window_frame = st.image([], width=200)

# while webcam_capture:
# 	# keep webcam connected
# 	_, frame = cam.read()
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 	# feature scan
# 	#video_scan(frame, st.session_state.classifiers)
# 	# show video
# 	window_frame.image(frame)
# else:
# 	cam.release()

# webrtc_streamer(key="example")
