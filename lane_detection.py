import cv2
import numpy as np

def lane_detection(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Apply Canny edge detection
  edges = cv2.Canny(blurred, 50, 150)

  # Define region of interest (ROI) for lane detection
  height, width = image.shape[:2]
  roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)],], dtype=np.int32)
  mask = np.zeros_like(edges)
  cv2.fillPoly(mask, roi_vertices, 255)
  masked_edges = cv2.bitwise_and(edges, mask)

  # Apply Hough transform to detect lines
  lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=60, minLineLength=40, maxLineGap=50)

  # Filter and fit lane lines
  left_lane = []
  right_lane = []
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      slope = (y2 - y1) / (x2 - x1)
      intercept = y1 - slope * x1
      if slope < -0.5:  # Left lane
        left_lane.append((slope, intercept))
      elif slope > 0.5:  # Right lane
        right_lane.append((slope, intercept))

  # Average lane lines (optional, improve robustness)
  if left_lane:
    left_lane_average = np.average(left_lane, axis=0)
  else:
    left_lane_average = None
  if right_lane:
    right_lane_average = np.average(right_lane, axis=0)
  else:
    right_lane_average = None

  # Draw lane lines (if detected)
  line_image = image.copy()
  if left_lane_average is not None:
    y1 = height
    y2 = height // 2
    x1 = int((y1 - left_lane_average[1]) / left_lane_average[0])
    x2 = int((y2 - left_lane_average[1]) / left_lane_average[0])
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)
  if right_lane_average is not None:
    y1 = height
    y2 = height // 2
    x1 = int((y1 - right_lane_average[1]) / right_lane_average[0])
    x2 = int((y2 - right_lane_average[1]) / right_lane_average[0])
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

  return line_image

# Read the image
image = cv2.imread('data/img-28-02-2024-19:30:49-snapshot-night-images-nb9.jpg')

# Perform lane detection
result_image = lane_detection(image)

# Display the original and result image
cv2.imshow('Original Image', image)
cv2.imshow('Lane Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()