
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL.Image import open as open_image
from urllib.request import urlopen

# Function to load pictures from a dict of urls
file_urls = {
    'lena.png':       'https://i.imgur.com/ncE3dty.png',
    'ednaldo.jpg':    'https://i.imgur.com/JnJD9FB.jpg',
    #'batman.jpg':     'https://i.imgur.com/8jKnbg5.jpg',
    #'calculista.jpg': 'https://i.imgur.com/MQNty8H.jpg',
    'vacilao.jpg':    'https://i.imgur.com/cgp0aY9.jpg',
    'harold.jpg':     'https://i.imgur.com/C8YrIjB.jpg',
    'tool.jpg':       'https://i.imgur.com/bgyZxWt.jpg'
}
load_image = lambda x : np.array(open_image(urlopen(file_urls[x])))

# Convert HEX to RBG
hex_to_rgb = lambda hex : tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
# Convert RGB to HEX
rgb_to_hex = lambda rgb_tuple : '#%02x%02x%02x' % rgb_tuple

# Dected feature for a given classifier
def feature_detection(rgb_image, gray_image, classifier, 
                      scaleFactor=1.3, minNeighbors=5, sub_search=False,
                      color=(255,0,0), **kwargs):
    
    # Create empty tuples for roi
    roi_rgb, roi_gray = (), ()
    
    # Select region of interest
    classifier_results = classifier.detectMultiScale(gray_image, scaleFactor, int(minNeighbors))
        
    # Draw rectangle for current feature
    for (cx, cy, cw, ch) in classifier_results:
        
        # Apply drawing in image
        cv2.rectangle(img=rgb_image,       # image to draw
                      pt1=(cx, cy),        # top-left corner
                      pt2=(cx+cw, cy+ch),  # bottom-right corner
                      color=color,         # color of the rectangle (red)
                      thickness=2)         # thickness of the rectangle
    
        # Select the region of interest, both for gray and rgb scale
        if sub_search:
            roi_rgb  += ( rgb_image[cy:cy+ch, cx:cx+cw],)
            roi_gray += (gray_image[cy:cy+ch, cx:cx+cw],) 

    # Return roi values
    return roi_rgb, roi_gray

# Search for features given a dict of classifiers
def video_scan(frame, classifiers):
    
    # Create gray scale image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for classifier in classifiers.values():
        # Detect features (global)
        if not classifier['is_sub']:
            c_roi_rgb, c_roi_gray = feature_detection(frame, gray_frame, **classifier)
        # Detect sub_features
        if classifier['sub_search']:
            for roi_rgb, roi_gray in zip(c_roi_rgb, c_roi_gray):
                for sub_classifier in classifier['sub_class']:
                    feature_detection(roi_rgb, roi_gray, **classifiers[sub_classifier])


# Print FPS on frame
def print_fps(video, frame, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255)):
    
    # Get FPS
    fps = int(video.get(5))
    
    height, width, channel = frame.shape
    
    offset = int(height / len(str(fps))) - 10

    cv2.putText(img=frame, 
                text=str(fps), 
                org=(50, offset), 
                font=font, 
                fontScale=1, 
                color=color, 
                thickness=5,
                lineType=8)


