#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def webcam_publisher():
    # Initialize the ROS Node
    rospy.init_node('webcam_publisher', anonymous=True)

    # Create a publisher object
    pub = rospy.Publisher('/webcam/image', Image, queue_size=10)

    # Create a CvBridge object
    bridge = CvBridge()

    # Open the webcam device
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    # Set a reasonable publishing rate
    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            try:
                # Convert the OpenCV image to a ROS Image message
                ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")

                # Publish the image
                pub.publish(ros_image)
            except CvBridgeError as e:
                rospy.logerr(e)

        rate.sleep()

    # When everything done, release the capture
    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass