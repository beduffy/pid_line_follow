# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage, Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge
import cv2
import numpy as np
# from simple_pid import PID
from shapely.geometry import Polygon


def use_chosen_contour_for_pid_set_point(chosen_contour):
    if chosen_contour is not None:
        y_positions = chosen_contour[:, :, 1].flatten()
        sorted_indices = np.argsort(y_positions)
        one_third_index = sorted_indices[len(sorted_indices) // 3]
        one_third_point = chosen_contour[one_third_index][0]
        cX, cY = one_third_point[0], one_third_point[1]
        error = cX - image_width // 2

        correction = 0
        # raw_correction = pid(error)
        # Normalize correction to be proportional to linear speed, ensuring it's within a suitable range for PID control
        # correction = np.tanh(raw_correction) * 0.05
        # correction = raw_correction * 0.02

        # print(f"Centroid: ({cX}, {cY}), Error: {error}, raw_correction: {raw_correction}, Correction: {correction}")
        return correction, error, cX, cY
    else:
        print("No contour chosen, skipping PID control.")
        return None, None, None, None

import os
import glob

# Define the path to the folder containing the images
image_folder_path = 'data/march_27/'
image_folder_path = 'data/march_27_tuned_gucv'
# Use glob to find all jpg files in the folder
image_files = glob.glob(os.path.join(image_folder_path, '*.jpg'))

for image_file in image_files:
    image = cv2.imread(image_file)
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    image_width = w  # Set the image width for PID calculations

    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred_gray, 30, 100, apertureSize=3)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()

    center_x = w // 2
    image_third_height = h * 2 // 3  # Calculate the height which divides the image into bottom two thirds
    min_contour_points = 5  # Minimum number of points for a contour to be considered
    chosen_contour = None
    max_yellow_nearby = 0  # Initialize maximum yellow nearby count

    def nothing(x):
        pass

    def onTrack1(val):
        global hueLow
        hueLow = val
        print('Hue Low', hueLow)


    def onTrack2(val):
        global hueHigh
        hueHigh = val
        print('Hue High', hueHigh)


    def onTrack3(val):
        global satLow
        satLow = val
        print('Sat Low', satLow)


    def onTrack4(val):
        global satHigh
        satHigh = val
        print('Sat High', satHigh)


    def onTrack5(val):
        global valLow
        valLow = val
        print('Val Low', valLow)


    def onTrack6(val):
        global valHigh
        valHigh = val
        print('Val High', valHigh)

    hueLow = 0
    hueHigh = 179
    satLow = 27
    satHigh = 255
    valLow = 200
    valHigh = 226

    # Convert image to HSV color space to better identify yellow color
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a window for the trackbars
    cv2.namedWindow('myTracker')

    cv2.createTrackbar('Hue Low', 'myTracker', hueLow, 179, onTrack1)
    cv2.createTrackbar('Hue High', 'myTracker', hueHigh, 179, onTrack2)
    cv2.createTrackbar('Sat Low', 'myTracker', satLow, 255, onTrack3)
    cv2.createTrackbar('Sat High', 'myTracker', satHigh, 255, onTrack4)
    cv2.createTrackbar('Val Low', 'myTracker', valLow, 255, onTrack5)
    cv2.createTrackbar('Val High', 'myTracker', valHigh, 255, onTrack6)

    cv2.imshow('image', image)
    lower_yellow = np.array([hueLow, satLow, valLow])
    upper_yellow = np.array([hueHigh, satHigh, valHigh])

    def tune_hsv_values_with_trackbar_in_loop():
        while True:
            # Get the current positions of the trackbars
            hueLow = cv2.getTrackbarPos('Hue Low', 'myTracker')
            hueHigh = cv2.getTrackbarPos('Hue High', 'myTracker')
            satLow = cv2.getTrackbarPos('Sat Low', 'myTracker')
            satHigh = cv2.getTrackbarPos('Sat High', 'myTracker')
            valLow = cv2.getTrackbarPos('Val Low', 'myTracker')
            valHigh = cv2.getTrackbarPos('Val High', 'myTracker')

            # Define range for yellow color in HSV using trackbar values
            lower_yellow = np.array([hueLow, satLow, valLow])
            upper_yellow = np.array([hueHigh, satHigh, valHigh])

            # Create a mask for yellow color
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # Display the yellow mask
            cv2.imshow('Yellow Mask', yellow_mask)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    
    tune_hsv_values_with_trackbar_in_loop()

    # yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    if contours:
        for contour in contours:
            # Calculate the y-coordinates of the contour to check if most points are in the bottom two thirds
            _, y, _, height = cv2.boundingRect(contour)
            if y + height < image_third_height or len(contour) < min_contour_points:
                continue
            # Create a mask from the current contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            # Calculate the amount of yellow within the contour
            yellow_within_contour = cv2.bitwise_and(yellow_mask, yellow_mask, mask=contour_mask)
            yellow_count = cv2.countNonZero(yellow_within_contour)
            # Update chosen_contour if the current contour has more yellow nearby
            if yellow_count > max_yellow_nearby:
                max_yellow_nearby = yellow_count
                chosen_contour = contour
    if chosen_contour is not None:
        cv2.drawContours(image_with_contours, [chosen_contour], -1, (0, 255, 0), 3)
        y_positions = chosen_contour[:, :, 1].flatten()
        lowest_y = np.min(y_positions)
        highest_y = np.max(y_positions)
        print(f"Lowest Y: {lowest_y}, Highest Y: {highest_y}")
    for contour in contours:
        if not np.array_equal(contour, chosen_contour):
            cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 255), 3)

    # Use the chosen contour to calculate PID and publish cmd_vel
    correction, error, cX, cY = use_chosen_contour_for_pid_set_point(chosen_contour)

    if chosen_contour is not None and correction is not None:
        # Calculate the direction vector based on the correction
        direction = -np.sign(correction)
        arrow_length = 50
        arrow_thickness = 2
        arrow_color = (255, 0, 0)  # Red color for the direction arrow

        # Calculate start and end points for the arrow
        start_point = (cX, cY)
        end_point = (int(cX + direction * arrow_length), cY)

        # Draw the direction arrow on the image
        cv2.arrowedLine(image_with_contours, start_point, end_point, arrow_color, arrow_thickness)

        # Display the direction information on the image
        direction_text = "Left" if direction < 0 else "Right"
        cv2.putText(image_with_contours, f"Direction: {direction_text}, correction: {correction:.5f}, error: {error:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
    cv2.imshow('image_with_contours', image_with_contours)
    cv2.waitKey(0)
cv2.destroyAllWindows()

