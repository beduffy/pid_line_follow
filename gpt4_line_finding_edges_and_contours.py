import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image from file
image_path = 'data/imageedit_2_8049355487.png'
image = cv2.imread(image_path)

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
    for contour_idx in range(contour.shape[0]):
        if contour[contour_idx][0][0] > h / 2:
            contour_is_in_bottom_half = True
    if contour_is_in_bottom_half:
        cropped_image_with_contours = image_with_contours.copy()
        cv2.drawContours(cropped_image_with_contours, [contour], -1, (0, 255, 0), 2)
        cv2.imshow('contour', cropped_image_with_contours)
        cv2.waitKey(0)

# Now let's put the cropped image back to the original image so we can see the result
# image_with_contours[h//2:h, 0:w] = cropped_image_with_contours

# Convert the image to RGB and show it using matplotlib
image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
plt.imshow(image_with_contours_rgb)
plt.axis('off')  # Turn off axis numbers
plt.show()

















