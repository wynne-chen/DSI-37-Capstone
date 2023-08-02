# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) DSI 37 Capstone Project - Muay Th.AI Trainer

### Problem Statement

The surging global popularity of Muay Thai has led to an influx of new practitioners. Proper form is essential to prevent injuries and optimise performance. However, group classes lack personalised attention from instructors to correct form. Costly personal training is not affordable for everyone, especially casual hobbyists. Thus, there is a market opportunity for an AI Muay Thai trainer capable of observing users and offering real-time form recommendations. This solution could enhance gym training, allowing practitioners of all levels to refine their technique and achieve optimal results.

This project seeks to build a test version that will address only two moves in Muay Thai: a jab and a roundhouse kick.

### Objectives

* The primary objective entails building a model that can clearly distinguish between two classes of action: a jab and a right roundhouse kick.

* The model then proceeds to count the number of reps of each class, as well as to give advice to improve the form of the user.



---

### Data Creation

The training dataset comprises data extracted by me using the Tensorflow MoveNet Thunder model. 

I filmed the videos over the course of 2-3 weeks in late July to early August 2023. 

I used a single iPhone 14 Pro on 0.5x zoom in a standing tripod, and filmed in landscape.

There are a total of 60 columns. 

The first column is the class, of which there are only two options: jab or kick. 

The following 51 consist of 3 values for each of the 17 keypoints listed above. The three values, in order, are the normalised y co-ordinate, the normalised x co-ordinate, and the confidence of the keypoint. To be further explained in the data dictionary. To avoid repetition, I will only explain the x, y, and confidence values once. The naming convention for these 51 features are bodypart_x, bodypart_y, and bodypart_conf. For example for the nose: nose_x, nose_y, nose_conf. 

The last 8 columns are features that I created. See the data dictionary for more information.

### Data dictionary:

|column| datatype|explanation|
|:-|:-:|:-|
|**class**|*string*| The name of the class. Pre-labelled based on the videos. Either `jab` or `kick`.|
|**bodypart_y**|*float*| The scaled y co-ordinate of the bodypart. The value is from 0 to 1.|
|**bodypart_x**|*float*| The scaled x co-ordinate of the bodypart. The value is from 0 to 1.|
|**bodypart_conf**|*float*| The confidence level of the keypoint. Meaning how confident MoveNet was that this is the actual location of the bodypart in question. The value is from 0 to 1.|
|**left_elbow_angle**|*float*| The angle of the left elbow, in degrees. Measured in relation to the left shoulder and wrist.|
|**left_hip_angle**|*float*| The angle of the left hip, in degrees. Measured in relation to the left shoulder and knee.|
|**left_knee_angle**|*float*| The angle of the left knee, in degrees. Measured in relation to the left hip and ankle.|
|**right_elbow_angle**|*float*| The angle of the right elbow, in degrees. Measured in relation to the right shoulder and wrist.|
|**right_hip_angle**|*float*| The angle of the right hip, in degrees. Measured in relation to the right shoulder and knee.|
|**right_knee_angle**|*float*| The angle of the right knee, in degrees. Measured in relation to the right hip and ankle.|
|**left_eye_left_wrist**|*float*| The scaled distance between the left eye and left wrist. The value is from 0 to 1.|
|**left_ankle_right_ankle**|*float*| The scaled distance between the left ankle and right ankle. The value is from 0 to 1.|

<br>


---

### Conclusion

The MoveNet Thunder model by the Google Brain team is extremely robust. With very little effort it was capable of creating keypoints, allowing me to swiftly build a model that easily distinguished between the two inital classes of `jab` and `kick` to a high degree of accuracy.

However, the model reacted poorly once it was taken out of the specific scenario it was trained for; once the angle or lighting changed, and once there was a mix of jabs and kicks in the same video, the model had an excessive rate of false positives. 

This was eventually mitigated through coding in a third `guard` class and retraining the model.

Overall, while the model is roughly successful in its aims, there are many areas for future improvement.

### Next steps

I would like to improve the speed of the model, and also to train it on different angles and lighting conditions. 

Beyond that, I would like to extend the model to encompass the basic moveset in Muay Thai:

- right cross, or right straight punch
- hooks (left and right)
- uppercuts (left and right)
- the teep, or push kick
- elbows (horizontal, upward, downward, backward)
- knees (left and right)

Beyond that, it would be good to train the model to recognise the southpaw stance as well. 

Ultimately, this model should be able to count the reps in a simple heavy bag workout session, as well as to give form corrections where necessary. 





