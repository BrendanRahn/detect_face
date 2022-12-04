#! /usr/bin/env python3

# rospy for the subscriber
import rospy

from sensor_msgs.msg import Image             # ROS image message

from cv_bridge import CvBridge, CvBridgeError # ROS image message to opencv2 image converter

import cv2 # opencv2 for image processing

import pdb
import pickle
import os

bridge = CvBridge()

image_counter  = 1
# Step 1 (TO DO): create a global counter for detected face
# YOUR CODE

detected_face = 1


def image_callback(msg):

    print("Received an image from the publisher!")

    try:

        # convert your ROS Image message to opencv2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")         

    except CvBridgeError as e:

        print(e)


    # YOUR CODE HERE FOR SAVING HISTOGRAM OF THE IMAGE AS PICKLE FILE
    global image_counter
    global detected_face
    # Step 2 (TO DO): access the global counter for detected face
    # YOUR CODE


    face_dir = "C:/Users/Brendan/Desktop/OneDrive - Drake University/(3) Robotics/ROS_workspace/src/locobot_project/faces/face_detected"

    file_name = "img_%04d"%(image_counter) + ".jpg"

    # ...

    # ----------- STEP 1: Load Viola-Jones Face Detector (pretrained model) -----------
    trained_model_name  = 'C:/Users/Brendan/Desktop/OneDrive - Drake University/(3) Robotics/ROS_workspace/src/locobot_project/src/haarcascades/haarcascade_frontalface_alt.xml'
    trained_model       = cv2.CascadeClassifier()
    trained_model.load(trained_model_name)



    # ----------- STEP 2: process the input image -----------
    gray_scale_img      = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)    
    gray_scale_img      = cv2.equalizeHist(gray_scale_img)
    
    
    # ----------- STEP 3: apply Viola-Jones Face Detector to the input gray-scale image -----------
    try:

        faces           = trained_model.detectMultiScale(gray_scale_img)



        img_with_det    = gray_scale_img

        # ----------- STEP 4: save the detected portion of the image -----------
        for (x,y,w,h) in faces:
            
            top_left_point      = (x, y)
            bottom_right_point  = (x+w, y+h)
            color               = (200, 0, 0)
            thickness           = 5
            img_with_det        = cv2.rectangle(cv2_img, top_left_point, bottom_right_point, color, thickness)
            cv2_img_face             = gray_scale_img[y:y+h,x:x+w]

            print("found face: ", faces)
            # TO-DO: maintain a counter to keep track of the detected face
            # YOUR CODE


        # Step 4 (TO DO): save your image (color with rectangular bounding box)
        # YOUR CODE


        os.chdir(face_dir)
        cv2.imwrite(file_name, cv2_img_face)
        
        detected_face += 1
        image_counter += 1

    
    except:

        print("face not found: ")        
        # Step 5 (TO DO): save your image (gray-scale)
        # YOUR CODE
        
        cv2.imwrite(file_name, cv2_img)

    cv2.waitKey(2000) # WAIT FOR 2000 MILISECONDS


    image_counter   = image_counter + 1



    # Step 6 (TO DO): compute accuracy
    # YOUE CODE


    accuracy = detected_face/image_counter

    print("Face detection of accuracy: ", accuracy)

def start_subscribing():

    rospy.init_node('Node image subsriber: initialized ...')

    image_topic = '/machine_cam'        # topic under which the image will be subscribed
    

    # Subscriber
    # ----------- single subscriber ---------------
    rospy.Subscriber(image_topic, Image, image_callback) # # initializing your subscriber and defining the callback

    # spin until ctrl + c
    rospy.spin()



try:

    start_subscribing()

except rospy.ROSInterruptException:

    print("failed to call the start_subscribing ...")

