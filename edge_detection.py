import cv2
import numpy as np
import glob

def loop_through_images_in_folder(folder):
    image_paths = glob.glob('data/*jpg')
    print(image_paths)


    for image_path in image_paths:
        # TODO loop through all images
        # Load the image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # edges_on_non_blurred = cv2.Canny(gray)

        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 0, 100)

        # Display the original image and the edges
        cv2.imshow('Original Image', img)
        cv2.imshow('Edge Image on blurred', edges)
        cv2.moveWindow('Edge Image on blurred',650,300)
        # cv2.imshow('Edge Image on non blurred', edges_on_non_blurred)

        # Wait for a key press
        cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()


loop_through_images_in_folder('data')
# # TODO loop through all images
# # Load the image
# img = cv2.imread('data/img-28-02-2024-19:30:49-snapshot-night-images-nb9.jpg')

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # edges_on_non_blurred = cv2.Canny(gray)

# # Apply Canny edge detector
# edges = cv2.Canny(blurred, 0, 100)

# # Display the original image and the edges
# cv2.imshow('Original Image', img)
# cv2.imshow('Edge Image on blurred', edges)
# # cv2.imshow('Edge Image on non blurred', edges_on_non_blurred)

# # Wait for a key press
# cv2.waitKey(0)

# # Close all windows
# cv2.destroyAllWindows()