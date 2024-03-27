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


# image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
# image = cv2.imread('data/march_27/2024-03-27-16-30-59-800.jpg')
image = cv2.imread('data/march_27/2024-03-27-19-17-59-602318.jpg')
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
image_width = w  # Set the image width for PID calculations

# blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
# blurred_gray = gray
edges = cv2.Canny(blurred_gray, 30, 100, apertureSize=3)
edges = cv2.dilate(edges, None, iterations=1)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()

center_x = w // 2
image_third_height = h * 2 // 3  # Calculate the height which divides the image into bottom two thirds
min_contour_points = 5  # Minimum number of points for a contour to be considered
chosen_contour = None
if contours:
    chosen_contour_pair = None  # Initialize with None to handle case where no contour meets criteria
    max_intersection = 0  # Initialize maximum intersection area
    for i, contour1 in enumerate(contours):
        # Calculate the y-coordinates of the contour to check if most points are in the bottom two thirds
        _, y1, _, height1 = cv2.boundingRect(contour1)
        contour1_bottom_thirds_points = sum(y1 + point[0][1] > image_third_height for point in contour1)
        if len(contour1) < min_contour_points or contour1_bottom_thirds_points/len(contour1) < 0.5:
            continue
        for j, contour2 in enumerate(contours[i+1:], start=i+1):
            # Calculate the y-coordinates of the contour to check if most points are in the bottom two thirds
            _, y2, _, height2 = cv2.boundingRect(contour2)
            contour2_bottom_thirds_points = sum(y2 + point[0][1] > image_third_height for point in contour2)
            if len(contour2) < min_contour_points or contour2_bottom_thirds_points/len(contour2) < 0.5:
                continue
            # Calculate the bounding rotated rectangles for each contour
            rect1 = cv2.minAreaRect(contour1)
            box1 = cv2.boxPoints(rect1)
            box1 = np.int0(box1)
            rect2 = cv2.minAreaRect(contour2)
            box2 = cv2.boxPoints(rect2)
            box2 = np.int0(box2)
            # Convert boxes to polygons
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            # Calculate intersection area
            intersection_area = poly1.intersection(poly2).area
            # Update chosen_contour_pair if the current pair has a larger intersection area
            if intersection_area > max_intersection:
                max_intersection = intersection_area
                chosen_contour_pair = (contour1, contour2)
    chosen_contour = chosen_contour_pair[0] if chosen_contour_pair else None  # Assign the chosen pair to chosen_contour for further processing

for contour in contours:
    if np.array_equal(contour, chosen_contour):
        cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)
        y_positions = contour[:, :, 1].flatten()
        lowest_y = np.min(y_positions)
        highest_y = np.max(y_positions)
        print(f"Lowest Y: {lowest_y}, Highest Y: {highest_y}")
    else:
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

