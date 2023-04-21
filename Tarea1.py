import cv2
import numpy as np
import time

IMG_ROW_RES = 480
IMG_COL_RES = 640

def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3, IMG_COL_RES)
    ret = video_capture.set(4, IMG_ROW_RES)
    return video_capture

def acquire_image(video_capture):
    #grab a single frame of video
    ret, frame = video_capture.read()
    scaled_rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    scaled_rgb_frame = scaled_rgb_frame[:, :, ::-1]
    return frame, scaled_rgb_frame

def show_frame(name, frame):
    #Display the resulting image frame in the PAC
    cv2.imshow(name,frame)

def highlight_polygons(img_gray):
    # Detect edges using Canny
    edges = cv2.Canny(img_gray, 50, 200)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    img_out = np.zeros_like(img_gray)
    cv2.drawContours(img_out, contours, -1, 255, thickness=2)

    # Highlight the polygons
    highlighted_gray = cv2.addWeighted(img_gray, 0.5, img_out, 0.5, 0)

    return highlighted_gray


##############################################################################
#COMMUNICATION - LANGUAGE - PROTOCOLS
#Defining some time scales for communicatiions
lastPublication = 0.0
PUBLISH_TIME = 0.1
##############################################################################

##############################################################################
# SENSING - PERCEPTION
#Setup and initialization of perception
video_capture = init_camera()


##############################################################################
#Percetion-action-loop
##############################################################################
#While ALIVE DO
while (True):
    # Sensing layer
    bgr_frame, scaled_rgb = acquire_image(video_capture)
    ##############################################################################
    #COMMUNICATION LAYER: messages to trigger actions
    #on the external world (SPATIAL-temporal SCALES)

    if np.abs(time.time()-lastPublication) > PUBLISH_TIME:
        try:
            print("No remote action needed ...")
        except (KeyboardInterrupt):
            break
        except Exception as e:
            print (e)
        lastPublication = time.time()
    

    # Computation layer - visual processing pipeline
    img_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    highlighted_gray = highlight_polygons(img_gray)
    edges = cv2.Canny(img_gray, 50, 200)
    bgr_frame[edges == 255] = [0, 0, 255]
    

    # Visualization layer
    show_frame('RGB image', bgr_frame)
    show_frame('Gray level image', img_gray)
    show_frame('Highlighted polygons', highlighted_gray)

    # Hit 'q' on the keyboard to quit:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#Release the webcam
video_capture.release()
cv2.destroyAllWindows()