import cv2
import numpy as np
import glob

def plot_hough_lines(image, lines):
  """
  Plots all lines returned by cv2.HoughLinesP on top of the original image.

  Args:
      image: The original image (BGR format).
      lines: The output from cv2.HoughLinesP (numpy array).
  """
  # Check if any lines were found
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      # Draw the line segment on the original image
      cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue color, thickness 2
  return image

def loop_through_images_in_folder(folder):
    image_paths = glob.glob('data/*jpg')
    print(len(image_paths))


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

        # import pdb;pdb.set_trace()

        edges[:300] = 0  # because only below certain part of image is the road

        # Perform Hough Line Transform (adjust parameters as needed)
        rho=1 
        theta=np.pi/180
        threshold=100
        min_line_length=20 
        max_line_gap=10
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        # lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        # lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        # Plot all detected lines on the original image
        image_with_lines = plot_hough_lines(img.copy(), lines)

        # Display the image with lines
        cv2.imshow("Image with Lines", image_with_lines)


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