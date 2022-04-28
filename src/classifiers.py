import cv2
import urllib

# Load Face classifier
frontal_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Eye classifier
eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load Smile classifier
smile_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load Upper Body classifier
upperbody_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Load Profile Face classifier
profile_face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Load Hand classifier
urllib.request.urlretrieve('https://raw.githubusercontent.com/Aravindlivewire/Opencv/master/haarcascade/aGest.xml', filename='xml/aGest.xml')
hand_classifier = cv2.CascadeClassifier('xml/aGest.xml')


# Create classifiers dictionary
sub_classifiers = {
    'eye':
    {
        'classifier'  : eye_classifier,
        'minNeighbors': 8,
        'color'       : (0,0,255),
        'sub_search'  : False,
        'sub_class'   : None
    },
    'smile':
    {
        'classifier'  : smile_classifier,
        'minNeighbors': 10,
        'color'       : (0,255,0),
        'sub_search'  : False,
        'sub_class'   : None
    }
}

classifiers = {
    'frontal_face':
    {
        'classifier'  : frontal_face_classifier,
        'minNeighbors': 5,
        'color'       : (255,0,0),
        'sub_search'  : True,
        'sub_class'   : [
                            sub_classifiers['eye'],
                            sub_classifiers['smile']
                        ]
    },
    'profile_face':
    {
        'classifier'  : profile_face_classifier,
        'minNeighbors': 5,
        'color'       : (255,0,255),
        'sub_search'  : True,
        'sub_class'   : [
                            sub_classifiers['eye'],
                            sub_classifiers['smile']
                        ]
    },
    'hand':
    {
        'classifier'  : hand_classifier,
        'minNeighbors': 3,
        'color'       : (255,255,0),
        'sub_search'  : False,
        'sub_class'   : None
    },
}