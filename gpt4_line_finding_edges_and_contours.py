import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# Let's attempt the process again, ensuring that we handle the mask correctly.

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Use Canny edge detection to find edges
# edges = cv2.Canny(gray, 0, 100, apertureSize=3)

# # Find contours from the edges, no dilation this time to get the precise lines
# contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # We will use a white mask to identify if a contour is white or not
# white_mask = np.zeros_like(gray)

# # Draw the contours on the mask with white color
# cv2.drawContours(white_mask, contours, -1, (255), 1)

# # Now, we create a new mask where we will draw only the contours that have white pixels on the gray image
# final_mask = np.zeros_like(gray)

# # We iterate through each contour and check if there are white pixels within the contour
# for contour in contours:
#     # Create a mask for the current contour
#     contour_mask = np.zeros_like(gray)
#     cv2.drawContours(contour_mask, [contour], -1, (255), -1)
    
#     # Calculate the mean color of the contour area in the original (gray) image
#     mean_val = cv2.mean(gray, mask=contour_mask)[0]
    
#     # If the mean value is high, it indicates that the contour area is white
#     # if mean_val > 200:
#     if mean_val > 10:
#         cv2.drawContours(final_mask, [contour], -1, (255), 1)

# # Apply the final mask to the original image
# image_with_white_lines = cv2.bitwise_and(image, image, mask=final_mask)
# cv2.imshow('Edge Image on blurred', edges)
# cv2.moveWindow('Edge Image on blurred',650,300)
# # cv2.imshow('Edge Image on non blurred', edges_on_non_blurred)

# # Wait for a key press
# cv2.waitKey(0)

# # Convert the image to RGB and show it using matplotlib
# image_with_white_lines_rgb = cv2.cvtColor(image_with_white_lines, cv2.COLOR_BGR2RGB)
# plt.imshow(image_with_white_lines_rgb)
# plt.axis('off')  # Turn off axis numbers
# plt.show()


def is_contour_in_bottom_half(contour, image_height):
    """
    Check if all points of a contour are in the bottom half of the image.

    Parameters:
    - contour: The contour to check.
    - image_height: The height of the image.

    Returns:
    - True if all points are in the bottom half, False otherwise.
    """
    # Calculate the y-coordinate which divides the image into a top and bottom half
    mid_height = image_height // 2

    # Check each point in the contour
    for point in contour:
        x, y = point[0]  # Extract x, y coordinates
        if y < mid_height:  # If the y-coordinate is in the top half, return False
            return False
    # If all points are in the bottom half, return True
    return True


# def is_line_like(contour, threshold=0.01):
#     # Approximate contour to reduce the number of points
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
    
#     # Calculate the contour's bounding rectangle
#     rect = cv2.minAreaRect(approx)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
    
#     # Calculate the area of the contour and the area of the bounding rectangle
#     contour_area = cv2.contourArea(approx)
#     rect_area = cv2.contourArea(box)
    
#     # Calculate the extent ratio (contour area / bounding rectangle area)
#     extent = contour_area / rect_area if rect_area != 0 else 0
    
#     # Check if the contour is line-like based on the extent ratio
#     return extent > threshold


import math

def is_line_like(contour, length_threshold=0.9):
    # Approximate contour to reduce the number of points
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Get the bounding rectangle of the approximated contour
    x, y, w, h = cv2.boundingRect(approx)

    # Calculate the diagonal length of the bounding rectangle
    rect_diagonal_length = math.sqrt(w**2 + h**2)

    # Calculate the length of the contour (perimeter)
    contour_length = cv2.arcLength(approx, True)
    
    # Calculate the ratio of contour length to bounding rectangle diagonal length
    length_ratio = contour_length / rect_diagonal_length if rect_diagonal_length != 0 else 0
    
    # Check if the contour is line-like based on the length ratio
    return length_ratio > length_threshold




def find_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

image_paths = glob.glob('data/*jpg')
print(len(image_paths))

# Load the image from file
# image_path = 'data/imageedit_2_8049355487.png'
for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path)

    # Alright, let's simply find the contours from the edges without considering the color.

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use Canny edge detection to find edges
    # edges = cv2.Canny(gray, 0, 100, apertureSize=3)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours from the Canny edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # We will draw the contours on the bottom half of the original image for visualization
    image_with_contours = image.copy()
    # cropped_image_with_contours = image_with_contours[h//2:h, 0:w]
    # cropped_image_with_contours = image_with_contours

    # Iterate through the contours and draw them on the original image
    for i, contour in enumerate(contours):
        # Draw the contour with a unique color for each
        # import pdb;pdb.set_trace()
        contour_is_in_bottom_half = False
        # for point_idx in range(contour.shape[0]):
            # if contour[point_idx][0][1] > h / 2:  # checking everything, not good TODO 
            # # if contour[point_idx][0][1] > 300:  # checking everything, not good TODO 
            #     # x, y, not y, x
            #     contour_is_in_bottom_half = True
            #     print(contour[point_idx][0][1])
        
        contour_is_in_bottom_half = is_contour_in_bottom_half(contour, h)

        if contour_is_in_bottom_half and contour.shape[0] > 300:
            cropped_image_with_contours = image_with_contours.copy()
            print('Num contours: ', contour.shape[0])
            contour_is_line_like = is_line_like(contour)
            print('contour_is_line_like:', contour_is_line_like)
            
            cv2.drawContours(cropped_image_with_contours, [contour], -1, (0, 255, 0), 2)
            cv2.imshow('contour', cropped_image_with_contours)
            cv2.waitKey(0)
            # cv2.waitKey(100)  # to have nice view of all quick
            # import pdb;pdb.set_trace()

    # Assuming the image's width is w
    image_center_x = w // 2

    # Find the contour with the centroid closest to the center of the image
    closest_to_center = None
    min_distance_to_center = float('inf')

    for cnt in contours:
        centroid = find_centroid(cnt)
        if centroid:
            distance_to_center = abs(centroid[0] - image_center_x)
            if distance_to_center < min_distance_to_center:
                closest_to_center = cnt
                min_distance_to_center = distance_to_center

    cv2.drawContours(image.copy(), [closest_to_center], -1, (0, 255, 0), 2)
    cv2.imshow('contour center', cropped_image_with_contours)
    cv2.waitKey(0)

    # Now let's put the cropped image back to the original image so we can see the result
    # image_with_contours[h//2:h, 0:w] = cropped_image_with_contours

    # Convert the image to RGB and show it using matplotlib
    # image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_with_contours_rgb)
    # plt.axis('off')  # Turn off axis numbers
    # plt.show()

















