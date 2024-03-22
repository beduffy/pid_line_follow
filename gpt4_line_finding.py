import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image from file
image_path = 'data/imageedit_2_8049355487.png'
image = cv2.imread(image_path)

# We need to rework the method to check for white lines.
# Let's first define a range for white color in HSV space, then apply this mask to the image to isolate white areas.

# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of white color in HSV
# Note: this range might need adjustment depending on the image lighting and quality
lower_white = np.array([0, 0, 168], dtype=np.uint8)
upper_white = np.array([172, 111, 255], dtype=np.uint8)

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image,image, mask= mask)

# Now let's find the contours on the mask for the white color
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0,255,0), 2)

# Convert the image to RGB and show it using matplotlib
image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
plt.imshow(image_with_contours_rgb)
plt.axis('off')  # Turn off axis numbers
plt.show()
