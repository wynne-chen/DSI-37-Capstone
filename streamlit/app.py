import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path


import os
import tempfile

import joblib
import pandas as pd
import numpy as np



import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import tensorflow as tf
import cv2




# import datasets

# import dataset 
final_path = Path(__file__).parent / 'data/combined_coords.csv'
df = pd.read_csv(final_path)

# import audio files
audio_path = Path(__file__).parent / 'audio_clips/'


# import sample videos
# jab 
jab_path = Path(__file__).parent / 'videos/sample_jab.mp4'
jab_video = open(jab_path, 'rb')
jab_bytes = jab_video.read()

# kick 
kick_path = Path(__file__).parent / 'videos/sample_kick.mp4'
kick_video = open(kick_path, 'rb')
kick_bytes = kick_video.read()


# load the model
model_path = Path(__file__).parent / 'models/rgf_muaythai.pkl'

with open(model_path, 'rb') as f:
    model = joblib.load(f)

# load the MoveNet model
interpreter = tf.lite.Interpreter(model_path = 'models/lite-model_movenet_singlepose_thunder_3.tflite')
input_size = 256
interpreter.allocate_tensors()




# dictionaries that are important

# keypoint dictionary
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
    }

# colour dictionary

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
    }


# important functions now

# visualisation functions

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.25):
    """
    Returns high confidence keypoints and edges for visualisation.
    
    Args:
    
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
                            the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be visualised.
    
    Returns:
    
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
      
    """

    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
            kpts_scores[edge_pair[1]] > keypoint_threshold):
            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
    """
    
    Draws the keypoint predictions on image.
    
    Args:
    
    image: A numpy array with shape [height, width, channel] representing the
            pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
                            the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
                  of the crop region in normalized coordinates (see the init_crop_region
                  function below for more detail). If provided, this function will also
                  draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
                          Note that the image aspect ratio will be the same as the input image.

    Returns:

    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
    """

    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))

    # To remove the huge white borders

    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)

    # Turn off tick labels

    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
                                                    keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)

    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)
    
    if crop_region is not None:

        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        
        ax.add_patch(rect)


    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC)
    
    return image_from_plot

# movenet function

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


# feature engineering functions
def calculate_angle(a,b,c):
    
    '''
    Input: three sets of (x,y) coordinates (3 tuples)
    Output: angle of joint in degrees (1 float)
    '''
    
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def joint_coords(df, bodypart):
    cols = [x for x in df.columns if bodypart in x]
    joint_coords = zip(df[cols[1]], df[cols[0]])
    return list(joint_coords)

def all_angles(coords_listA, coords_listB, coords_listC):
    angles_list = []
    for (a, b, c) in zip(coords_listA, coords_listB, coords_listC):
        angles_list.append(calculate_angle(a,b,c))
    return angles_list

def distance(a, b):
    a = np.array(a) 
    b = np.array(b) 
    
    dist = ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    return dist

def all_dist(coords_listA, coords_listB):
    dist_list = []
    for (a, b) in zip(coords_listA, coords_listB):
        dist_list.append(distance(a,b))
    return dist_list

def new_features(df):   
    
    # define (x,y) coordinates for relevant bodyparts
    
    # left
    left_shoulder = joint_coords(df, 'left_shoulder')
    left_elbow = joint_coords(df, 'left_elbow')
    left_wrist = joint_coords(df, 'left_wrist')
    left_hip = joint_coords(df, 'left_hip')
    left_knee = joint_coords(df, 'left_knee')
    left_ankle = joint_coords(df, 'left_ankle')
    left_eye = joint_coords(df, 'left_eye')
    
    # right
    right_shoulder = joint_coords(df, 'right_shoulder')
    right_elbow = joint_coords(df, 'right_elbow')
    right_wrist = joint_coords(df, 'right_wrist')
    right_hip = joint_coords(df, 'right_hip')
    right_knee = joint_coords(df, 'right_knee')
    right_ankle = joint_coords(df, 'right_ankle')
    right_eye = joint_coords(df, 'right_eye')
    
    # add new columns with the new angles 
    df['left_elbow_angle'] = all_angles(left_shoulder, left_elbow, left_wrist)
    df['left_hip_angle'] = all_angles(left_shoulder, left_hip, left_knee)
    df['left_knee_angle'] = all_angles(left_hip, left_knee, left_ankle)

    df['right_elbow_angle'] = all_angles(right_shoulder, right_elbow, right_wrist)
    df['right_hip_angle'] = all_angles(right_shoulder, right_hip, right_knee)
    df['right_knee_angle'] = all_angles(right_hip, right_knee, right_ankle)
    
    # add new columns with the new distances
    df['left_eye_left_wrist'] = all_dist(left_eye, left_wrist)
    df['right_eye_right_wrist'] = all_dist(right_eye, right_wrist)
    df['left_ankle_right_ankle'] = all_dist(left_ankle, right_ankle)
    
    
    return(df)



# key definition

columns = []
for key, value in KEYPOINT_DICT.items():
    columns.extend([key + '_y', key + '_x', key + '_conf'])


    





# now for the actual app




# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='Muay Th.AI Trainer',
    page_icon=':boxing_glove:',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['About','Live Muay Th.AI', 'Upload A Video'],
    icons = ['eyeglasses','camera-reels-fill','collection-play-fill'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#FF7F0E'}
        }
    )

if selected == 'About':
    # title
    st.title('About')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)



    # Introduction to the whole project
    
    st.subheader('What is this Project?')
    
    st.write('This is a Muay Thai Trainer app that I built to help people who want some critique on their form.')
    st.write('Currently, this model is only trained on two classes: the jab and the right roundhouse kick. Also, this model is presently only trained on the orthodox stance.')
    st.write('You can keep reading this page for an explanation about what makes a good jab or kick and some differences between them.')
    st.write('Or, you can navigate to the Live Muay Th.AI Trainer in the second tab or the Video Critique Trainer in the third tab.')
    
    
    st.divider()
    
    # Explanation on the 2 classes
    # Beginning with the jab
    st.subheader('What is a Jab?')
    
    st.write("The Muay Thai jab is a quick, straight punch using the lead (or front) hand. It is a versatile and effective technique used to maintain distance, set up combinations, and gauge an opponent's reaction. The jab plays a crucial role in both offense and defense strategies in Muay Thai. In the orthodox stance, the jab is a punch thrown with the left hand.")
    
    st.video(jab_bytes)
    
    st.markdown(
    """
    These are some basic hallmarks of a good jab:
    - Lead arm reaches full extension before recoiling
    - Your head does not go past your feet (You dont lean forwards as you jab)
    - You return to your guard posture immediately after your jab
    """
    )
    
    st.divider()
        
    st.subheader('What is a Roundhouse Kick?')
    
    st.write("The Muay Thai roundhouse kick is a powerful kicking technique that involves pivoting on the supporting foot while swinging the other leg in a circular motion. It generates tremendous force from the hips, allowing the shin to strike the opponent, making it one of the sport's most devastating and signature moves.")
    
    st.video(kick_bytes)
    
    st.markdown(
    """
    These are some basic hallmarks of a good right roundhouse kick:
    - Your right arm swings down and your left hand swings up as you kick
    - You pivot on your standing (left) foot and turn your hips over fully
    - You return to your guard posture immediately after
    """
    )
    
    st.divider()    
    
    st.subheader('Differences In Movement')
    
    st.write('You can take a look at how different body parts move in he jab vs the kick')
    
    # comparative scatter plots
    # ask for user input
    
    # ask if they want to see it for the jab or kick
    j_or_k = st.radio('Which move would you like to study?',
                            ('Jab', 'Kick', 'Both'),
                            horizontal = True,
                            )
    
    # ask which keypoint they want to see the info for
    option = st.selectbox('Pick a body part to see how it moves',
                            ('Left Eye', 'Left Hand', 'Left Hip', 'Left Foot',
                             'Right Eye','Right Hand', 'Right Hip', 'Right Foot'))
    
    # translate the english from the option box into the equivalent variable
    if option == 'Left Eye':
        variable = 'left_eye'
    elif option == 'Left Hand':
        variable = 'left_wrist'
    elif option == 'Left Hip':
        variable = 'left_hip'
    elif option == 'Left Foot':
        variable = 'left_ankle'
    elif option == 'Right Eye':
        variable = 'right_eye'
    elif option == 'Right Hand':
        variable = 'right_wrist'
    elif option == 'Right Hip':
        variable = 'right_hip'
    elif option == 'Right Foot':
        variable = 'right_ankle'
        
    


    # create the dataframes for the plots    
    j = df[df['class']=='jab']
    k = df[df['class']=='kick']
    
    # draw the line graph for the chosen variable
    fig, ax = plt.subplots(figsize = (3,3))

    if j_or_k == 'Jab' or j_or_k == 'Both':
        sns.scatterplot(x = j[j[variable + '_conf']>0.2][variable + '_x'], y = 1 -j[j[variable + '_conf']>0.2][variable + '_y'], alpha = 0.2, color = 'g', label = 'jabs')
    if j_or_k == 'Kick' or j_or_k == 'Both':
        sns.scatterplot(x = k[k[variable + '_conf']>0.2][variable + '_x'], y = 1 - k[k[variable + '_conf']>0.2][variable + '_y'], alpha = 0.2, color = 'r', label = 'kicks')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.xlabel('X Co-ordinate', size=8)
    plt.ylabel('Y Co-ordinate', size=8)
    plt.title(str(option) + ' Co-ordinates in Both Classes (Confidence > 0.2)')
    plt.legend(loc='upper right')
    
    
    st.pyplot(fig)
    
    
    # now to draw the bar charts for confidence
    
    fig2, ax = plt.subplots(figsize = (3,3))

    if j_or_k == 'Jab' or j_or_k == 'Both':
        plt.hist(j[variable + '_conf'], bins = 10, alpha = 0.4, color = 'g', label = j['class'])
    if j_or_k == 'Kick' or j_or_k == 'Both':
        plt.hist(k[variable + '_conf'], bins = 10, alpha = 0.4, color = 'r', label = k['class'])
    
    plt.xlabel('Confidence', size=8)
    plt.ylabel('Frequency', size=8)
    
    if j_or_k == 'Jab':
        plt.title('Distribution of the ' + str(option) + ' Confidence in the Jab')
    elif j_or_k == 'Jab':
        plt.title('Distribution of the ' + str(option) + ' Confidence in the Kick')
    else:
        plt.title('Distribution of the ' + str(option) + ' Confidence in Both Classes')
        
        
    plt.legend(loc='upper right')

    
    st.pyplot(fig2)
    
    # section 2: feature engineered EDA
    
    st.subheader('Joint Angles And Distances')
    
    st.write('Besides studying individual body parts, I also took a look at some other features, such as joint angles and the relative distance between two body parts.')
    st.write('I selected these angles or relative distances based on my domain knowledge of what makes a good jab or roundhouse kick, as discussed earlier.')
    
    
    # ask if they want to see it for the jab or kick
    j_or_k2 = st.radio('Which move would you like to study?',
                            ('Jab', 'Kick', 'Both'),
                            horizontal = True,
                            key = 'section2'
                            )
    
    # ask which trait they want to see the info for
    option = st.selectbox('Pick a joint angle or relative distance',
                            ('Left Elbow Angle', 'Left Hip Angle', 'Left Knee Angle',
                             'Right Elbow Angle','Right Hip Angle', 'Right Knee Angle', 
                             'Distance Between Feet', 'Distance Between the Left Hand And Eyes'))
    
    # translate the english from the option box into the equivalent variable
    if option == 'Left Elbow Angle':
        variable = 'left_elbow_angle'
    elif option == 'Left Hip Angle':
        variable = 'left_hip_angle'
    elif option == 'Left Knee Angle':
        variable = 'left_knee_angle'
    elif option == 'Right Elbow Angle':
        variable = 'right_elbow_angle'
    elif option == 'Right Hip Angle':
        variable = 'right_hip_angle'
    elif option == 'Right Knee Angle':
        variable = 'right_knee_angle'
    elif option == 'Distance Between Feet':
        variable = 'left_ankle_right_ankle'
    elif option == 'Distance Between the Left Hand And Eyes':
        variable = 'left_eye_left_wrist'
    
    
    fig3, ax = plt.subplots(figsize = (3,3))

    if j_or_k2 == 'Jab' or j_or_k2 == 'Both':
        plt.hist(j[variable], bins = 10, alpha = 0.4, color = 'g', label = j['class'])
    if j_or_k2 == 'Kick' or j_or_k2 == 'Both':
        plt.hist(k[variable], bins = 10, alpha = 0.4, color = 'r', label = k['class'])
    
    if 'Angle' in option:
        plt.xlim(0,180)
        plt.xlabel('Angle(degrees)', size=8)
    else:
        plt.xlabel('Normalised Distance', size=8)
    
    plt.ylabel('Frequency', size=8)
    
    if j_or_k2 == 'Jab':
        plt.title('Distribution of the ' + str(option) + ' in the Jab')
    elif j_or_k2 == 'Jab':
        plt.title('Distribution of the ' + str(option) + ' in the Kick')
    else:
        plt.title('Distribution of the ' + str(option) + ' in Both Classes')
    
    
    plt.legend(loc='upper right')

    
    st.pyplot(fig3)
    
    
    st.subheader('')
    st.write('')
    st.write('')


if selected == 'Live Muay Th.AI':
    
    # title
    st.title('Live Muay Thai Training')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Use Your Device Camera To Film Your Workout')
    
    st.subheader('Instructions')
    
    st.markdown(
        """
        - Film yourself using a tripod placed perpendicular to yourself so the phone is filming your side view. Turn the phone so it is filming in landscape.
        - Ensure that both your feet are visible to the camera (you may want to use the 0.5x camera)
        - Download the results (Video will be slowed down)
        """
        )
    
    # first to set the stage for the video upload/download
    # create path for returning analysed video
    
    analysed_live_video = 'live_video_with_analysis.mp4'
    
    
    if os.path.exists(analysed_live_video):
        os.remove(analysed_live_video)
   
   
    # setting the stage for the counter/recommender
    stage = None
    jab_counter = 0
    kick_counter = 0
      

    cap = cv2.VideoCapture(1)

    # placeholder where the video will be once the start button is pressed
    frame_placeholder = st.empty()
    
    stop_button = st.button('Stop')
    
    # extracting the video info from the uploaded video for preparing the output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(analysed_live_video, fourcc, frame_fps, (width, height))
    
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
    
        if not ret:
            st.write('Video capture has ended.')
            break
        
        # Reshape image
        image = frame.copy()
        image_height, image_width, channels = image.shape
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    
    
        # Run model inference.
        keypoints_with_scores = movenet(input_image)
        
        
        # Make sure the camera is placed to the side 
        # and that the ankles can be seen
        # this will prevent the model from making inaccurate predictions it was not trained for
        
        # first, I will use the angle between the nose and shoulders to determine whether the camera
        # is to the side or not
        nose = (keypoints_with_scores[0][0][0][1], keypoints_with_scores[0][0][0][0])
        left_shoulder = (keypoints_with_scores[0][0][5][1], keypoints_with_scores[0][0][5][0])
        right_shoulder = (keypoints_with_scores[0][0][6][1], keypoints_with_scores[0][0][6][0])
        
        offset_angle = calculate_angle(left_shoulder, nose, right_shoulder)
        
        # next, to check the left angle confidence
        left_ankle_conf = keypoints_with_scores[0][0][15][2]
        right_ankle_conf = keypoints_with_scores[0][0][16][2]
        
        # now the command flow
        if (offset_angle > 45) and (left_ankle_conf < 0.3):
            # Tell user to move camera
            
            # Big box right in the middle of the screen, 1200 px by 300px
            cv2.rectangle(image, (360, 440), (1560, 740), (245, 117, 16), -1) 
            # top left corner, bottom right corner, colour, line thk (neg = filled)
    
            # Display warning telling them to move the camera
            cv2.putText(image, 'Please film yourself from the side'
                        , (410,540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, 'and ensure your ankles are visible'
                        , (410,640), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
        
        else:
            # Extract coordinates
            row = keypoints_with_scores[0][0].flatten().tolist()
    
            # Make Dataframe
            X = pd.DataFrame([row])
            X.columns = columns
            X = new_features(X)
            
            # Make Detections
            muay_thai_class = model.predict(X)[0]
            muay_thai_prob = model.predict_proba(X)[0]
            
    
    
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1) # top left corner, bottom right corner, colour, line thk (neg = filled)
    
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (145,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, muay_thai_class.split(' ')[0]
                        , (140,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
    
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (20,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(muay_thai_prob[np.argmax(muay_thai_prob)],2))
                        , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            
            
            
            # Jab/Kick counter and recommender logic
            
            # first to define the relevant parts
            left_eye = (keypoints_with_scores[0][0][1][1], keypoints_with_scores[0][0][1][0])
            left_wrist = (keypoints_with_scores[0][0][9][1], keypoints_with_scores[0][0][9][0])
            left_elbow = (keypoints_with_scores[0][0][7][1], keypoints_with_scores[0][0][7][0])
            left_hip = (keypoints_with_scores[0][0][11][1], keypoints_with_scores[0][0][11][0])
            left_ankle = (keypoints_with_scores[0][0][15][1], keypoints_with_scores[0][0][15][0])
            
            right_wrist = (keypoints_with_scores[0][0][10][1], keypoints_with_scores[0][0][10][0])
            right_elbow = (keypoints_with_scores[0][0][8][1], keypoints_with_scores[0][0][8][0])
            right_knee = (keypoints_with_scores[0][0][14][1], keypoints_with_scores[0][0][14][0])
            right_hip = (keypoints_with_scores[0][0][12][1], keypoints_with_scores[0][0][12][0])
            right_ankle = (keypoints_with_scores[0][0][16][1], keypoints_with_scores[0][0][16][0])
            
            # next, to define relevant distances and angles
            left_eye_left_wrist = distance(left_eye, left_wrist)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # I am redefining the right hip angle to be against the y axis
            # this is because using the (right_shoulder, right_hip, right_knee) definition gave bad results
            hip_y_axis = (right_hip[0], 0)
            right_hip_angle = calculate_angle(hip_y_axis, right_hip, right_knee)
            
            
            # prevent the model from running inferences if probability is low
            if muay_thai_prob[np.argmax(muay_thai_prob)] < 0.7:
                pass
            else:
                if muay_thai_class == 'guard':
    
                    # Jab/Kick counter logic
                    if stage == None:
                        stage = 'guard'
                        print(stage)
                    elif stage == 'jab':
                        stage = 'guard'
                        print(stage)
                        jab_counter += 1
                    elif stage == 'kick':
                        stage = 'guard'
                        print(stage)
                        kick_counter += 1
                        
                    # guard recommendation
                    if left_eye_left_wrist > 0.1:
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'KEEP YOUR GUARD UP'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            
                elif muay_thai_class == 'jab' and stage == 'guard':
                    stage = 'jab'
                    print(stage)
                    
                    # jab recommendations
                    if left_elbow_angle < 175:
                        print('Straighten your left arm')
                        
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'STRAIGHTEN YOUR LEFT ARM'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                        
                    elif left_eye[0] < left_ankle[0]:
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'STOP LEANING FORWARDS'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                        
                    else:
                        
                        # Box for advice in the top right of the screen, green because GOOD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (50, 255, 160), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'GOOD JAB'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                    
            
                elif muay_thai_class == 'kick' and stage == 'guard' and right_hip_angle < 100 and right_ankle_conf > 0.3:
                    stage = 'kick'
                    print(stage)
    
                    # kick recommendations
                    if right_elbow_angle < 110:
                        
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'SWING YOUR RIGHT ARM'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                    elif left_wrist[1] > left_eye[1]:
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'SWING YOUR LEFT ARM'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                   
                    elif right_hip_angle > 100:
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'KICK HIGHER'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                    
                    elif right_hip[0] > left_hip[0]:
                        # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'TURN YOUR HIPS OVER'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                    
                    else:
                        # Box for advice in the top right of the screen, green because GOOD, 500 x 60 px
                        cv2.rectangle(image, (1300, 60), (1800, 120), (50, 255, 160), -1) 
                        # top left corner, bottom right corner, colour, line thk (neg = filled)
                
                        # Display warning telling them to move the camera
                        cv2.putText(image, 'GOOD KICK'
                                    , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
    
                    
                
        # visualise the classes and probabilities and counts    
        
        # Get status box
        cv2.rectangle(image, (0,940), (250, 1080), (245, 117, 16), -1) # top left corner, bottom right corner, colour, line thk (neg = filled)
        
        # Jab Reps
        cv2.putText(image, 'JABS', (15,980), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(jab_counter), 
                    (30,1040), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
        # Kick Reps
        cv2.putText(image, 'KICKS', (125,980), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(kick_counter), 
                    (130,1040), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    
        # Visualise the predictions with image.
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
    
        
        
        # Display the processed video frames in real-time
        frame_placeholder.image(output_overlay, channels = 'BGR')
    
        video_output.write(output_overlay)
        
        if cv2.waitKey(10) & 0xFF==ord('q') or stop_button:
            break
    


    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()




    st.header('Final Counts:')
    st.subheader(f"Jabs : {jab_counter}")
    st.subheader(f"Kicks : {kick_counter}")
    

    
    




    if os.path.exists(analysed_live_video):
        with open(analysed_live_video, 'rb') as op_vid:
            st.download_button('Download Video', data = op_vid, file_name='live_video_with_analysis.mp4')
    
    
    
    



##---------------------------------------------------------------------##




if selected == 'Upload A Video':
    # title
    st.title('Form Advice On Video')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Upload A Past Session For Critique')
    
    st.subheader('Instructions')
    
    st.markdown(
        """
        - Film yourself using a tripod placed perpendicular to yourself so the phone is filming your side view. Turn the phone so it is filming in landscape.
        - Ensure that both your feet are visible to the camera (you may want to use the 0.5x camera)
        - Upload the video using the button below
        - Download the results (Video will be slowed down)
        """
        )
    
    # first to set the stage for the video upload/download
    # create path for returning analysed video
    
    analysed_video_file = 'video_with_analysis.mp4'
    
    
    if os.path.exists(analysed_video_file):
        os.remove(analysed_video_file)
    
    
    
    with st.form('Upload bagwork video here', clear_on_submit = True):
        video_data = st.file_uploader('Video Upload', type = ['mp4','mov', 'avi'])
        uploaded = st.form_submit_button("Upload")
    
    
    # placeholder where the video will be once uploaded
    frame_placeholder = st.empty()
    
    # setting the stage for the counter/recommender
    stage = None
    jab_counter = 0
    kick_counter = 0

    
    if video_data is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_data.read())

        cap = cv2.VideoCapture(temp_file.name)
        
        # extracting the video info from the uploaded video for preparing the output
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(analysed_video_file, fourcc, frame_fps, (width, height))
        
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            # Reshape image
            image = frame.copy()
            image_height, image_width, channels = image.shape
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        
        
            # Run model inference.
            keypoints_with_scores = movenet(input_image)
            
            
            # Make sure the camera is placed to the side 
            # and that the ankles can be seen
            # this will prevent the model from making inaccurate predictions it was not trained for
            
            # first, I will use the angle between the nose and shoulders to determine whether the camera
            # is to the side or not
            nose = (keypoints_with_scores[0][0][0][1], keypoints_with_scores[0][0][0][0])
            left_shoulder = (keypoints_with_scores[0][0][5][1], keypoints_with_scores[0][0][5][0])
            right_shoulder = (keypoints_with_scores[0][0][6][1], keypoints_with_scores[0][0][6][0])
            
            offset_angle = calculate_angle(left_shoulder, nose, right_shoulder)
            
            # next, to check the left angle confidence
            left_ankle_conf = keypoints_with_scores[0][0][15][2]
            right_ankle_conf = keypoints_with_scores[0][0][16][2]
            
            # now the command flow
            if (offset_angle > 45) and (left_ankle_conf < 0.3):
                # Tell user to move camera
                
                # Big box right in the middle of the screen, 1200 px by 300px
                cv2.rectangle(image, (360, 440), (1560, 740), (245, 117, 16), -1) 
                # top left corner, bottom right corner, colour, line thk (neg = filled)
        
                # Display warning telling them to move the camera
                cv2.putText(image, 'Please film yourself from the side'
                            , (410,540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'and ensure your ankles are visible'
                            , (410,640), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
            
            else:
                # Extract coordinates
                row = keypoints_with_scores[0][0].flatten().tolist()
        
                # Make Dataframe
                X = pd.DataFrame([row])
                X.columns = columns
                X = new_features(X)
                
                # Make Detections
                muay_thai_class = model.predict(X)[0]
                muay_thai_prob = model.predict_proba(X)[0]
                
        
        
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1) # top left corner, bottom right corner, colour, line thk (neg = filled)
        
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (145,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, muay_thai_class.split(' ')[0]
                            , (140,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (20,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(muay_thai_prob[np.argmax(muay_thai_prob)],2))
                            , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                
                
                
                # Jab/Kick counter and recommender logic
                
                # first to define the relevant parts
                left_eye = (keypoints_with_scores[0][0][1][1], keypoints_with_scores[0][0][1][0])
                left_wrist = (keypoints_with_scores[0][0][9][1], keypoints_with_scores[0][0][9][0])
                left_elbow = (keypoints_with_scores[0][0][7][1], keypoints_with_scores[0][0][7][0])
                left_hip = (keypoints_with_scores[0][0][11][1], keypoints_with_scores[0][0][11][0])
                left_ankle = (keypoints_with_scores[0][0][15][1], keypoints_with_scores[0][0][15][0])
                
                right_wrist = (keypoints_with_scores[0][0][10][1], keypoints_with_scores[0][0][10][0])
                right_elbow = (keypoints_with_scores[0][0][8][1], keypoints_with_scores[0][0][8][0])
                right_knee = (keypoints_with_scores[0][0][14][1], keypoints_with_scores[0][0][14][0])
                right_hip = (keypoints_with_scores[0][0][12][1], keypoints_with_scores[0][0][12][0])
                right_ankle = (keypoints_with_scores[0][0][16][1], keypoints_with_scores[0][0][16][0])
                
                # next, to define relevant distances and angles
                left_eye_left_wrist = distance(left_eye, left_wrist)
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # I am redefining the right hip angle to be against the y axis
                # this is because using the (right_shoulder, right_hip, right_knee) definition gave bad results
                hip_y_axis = (right_hip[0], 0)
                right_hip_angle = calculate_angle(hip_y_axis, right_hip, right_knee)
                
                
                # prevent the model from running inferences if probability is low
                if muay_thai_prob[np.argmax(muay_thai_prob)] < 0.7:
                    pass
                else:
                    if muay_thai_class == 'guard':
        
                        # Jab/Kick counter logic
                        if stage == None:
                            stage = 'guard'
                            print(stage)
                        elif stage == 'jab':
                            stage = 'guard'
                            print(stage)
                            jab_counter += 1
                        elif stage == 'kick':
                            stage = 'guard'
                            print(stage)
                            kick_counter += 1
                            
                        # guard recommendation
                        if left_eye_left_wrist > 0.1:
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'KEEP YOUR GUARD UP'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                
                    elif muay_thai_class == 'jab' and stage == 'guard':
                        stage = 'jab'
                        print(stage)
                        
                        # jab recommendations
                        if left_elbow_angle < 175:
                            print('Straighten your left arm')
                            
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'STRAIGHTEN YOUR LEFT ARM'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                            
                            
                        elif left_eye[0] < left_ankle[0]:
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'STOP LEANING FORWARDS'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                            
                            
                        else:
                            
                            # Box for advice in the top right of the screen, green because GOOD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (50, 255, 160), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'GOOD JAB'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                
                    elif muay_thai_class == 'kick' and stage == 'guard' and right_hip_angle < 100 and right_ankle_conf > 0.3:
                        stage = 'kick'
                        print(stage)
        
                        # kick recommendations
                        if right_elbow_angle < 110:
                            
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'SWING YOUR RIGHT ARM'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                            
                        elif left_wrist[1] > left_eye[1]:
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'SWING YOUR LEFT ARM'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                       
                        elif right_hip_angle > 100:
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'KICK HIGHER'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                        elif right_hip[0] > left_hip[0]:
                            # Box for advice in the top right of the screen, red because BAD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (0, 20, 255), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'TURN YOUR HIPS OVER'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
                        
                        else:
                            # Box for advice in the top right of the screen, green because GOOD, 500 x 60 px
                            cv2.rectangle(image, (1300, 60), (1800, 120), (50, 255, 160), -1) 
                            # top left corner, bottom right corner, colour, line thk (neg = filled)
                    
                            # Display warning telling them to move the camera
                            cv2.putText(image, 'GOOD KICK'
                                        , (1310,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
        
                        
                    
            # visualise the classes and probabilities and counts    
            
            # Get status box
            cv2.rectangle(image, (0,940), (250, 1080), (245, 117, 16), -1) # top left corner, bottom right corner, colour, line thk (neg = filled)
            
            # Jab Reps
            cv2.putText(image, 'JABS', (15,980), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(jab_counter), 
                        (30,1040), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
            # Kick Reps
            cv2.putText(image, 'KICKS', (125,980), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(kick_counter), 
                        (130,1040), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
            # Visualise the predictions with image.
            display_image = tf.expand_dims(image, axis=0)
            display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)
            output_overlay = draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)
        
            
            
            # Display the processed video frames in real-time
            frame_placeholder.image(output_overlay, channels = 'BGR')

            video_output.write(output_overlay)
        
           
        
        
        cap.release()
        video_output.release()
        temp_file.close()
        

        st.header('Final Counts:')
        st.subheader(f"Jabs : {jab_counter}")
        st.subheader(f"Kicks : {kick_counter}")
        
        
        
        
        if os.path.exists(analysed_video_file):
            with open(analysed_video_file, 'rb') as op_vid:
                st.download_button('Download Video', data = op_vid, file_name='video_with_analysis.mp4')


        
    


    
    
