import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from simple_pid import PID

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

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/webcam/image/compressed',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'webcam/processed_image', 10)
        self.edges_publisher_ = self.create_publisher(Image, 'webcam/edges_image', 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.pid = PID(0.1, 0.01, 0.01, setpoint=0)
        self.linear_speed = 0.2  # Assuming a default linear speed

    def listener_callback(self, data):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            self.image_width = w  # Set the image width for PID calculations

            blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred_gray, 55, 100, apertureSize=3)

            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            self.edges_publisher_.publish(edges_msg)

            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = image.copy()

            center_x = w // 2
            min_distance = float('inf')
            chosen_contour = None
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    distance = abs(cX - center_x)
                    if distance < min_distance:
                        min_distance = distance
                        chosen_contour = contour

            for contour in contours:
                if np.array_equal(contour, chosen_contour):
                    cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)
                    y_positions = contour[:, :, 1].flatten()
                    lowest_y = np.min(y_positions)
                    highest_y = np.max(y_positions)
                    print(f"Lowest Y: {lowest_y}, Highest Y: {highest_y}")
                else:
                    cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 255), 3)

            processed_image_msg = self.bridge.cv2_to_imgmsg(image_with_contours, encoding="passthrough")
            self.publisher_.publish(processed_image_msg)

            # Use the chosen contour to calculate PID and publish cmd_vel
            self.use_chosen_contour_for_pid_set_point(chosen_contour)
        except Exception as e:
            self.get_logger().error('Failed to process and publish image: %r' % (e,))

    def use_chosen_contour_for_pid_set_point(self, chosen_contour):
        if chosen_contour is not None:
            M = cv2.moments(chosen_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                error = cX - self.image_width // 2
                correction = self.pid(error)  # Changed from self.pid.update(error)
                twist_msg = Twist()
                twist_msg.linear.x = self.linear_speed
                twist_msg.angular.z = -correction
                self.cmd_vel_publisher_.publish(twist_msg)
                self.get_logger().info(f"Centroid: ({cX}, {cY}), Error: {error}, Correction: {correction}")
            else:
                self.get_logger().warn("Chosen contour has zero area, skipping PID control.")
        else:
            self.get_logger().warn("No contour chosen, skipping PID control.")
        

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()