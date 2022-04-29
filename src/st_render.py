
######################################################
#             Streamlit Rendering Functions
######################################################

import streamlit as st
from src.functions import *


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


