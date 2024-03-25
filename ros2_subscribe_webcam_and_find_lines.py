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
        # self.cmd_vel_publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.bridge = CvBridge()
        self.pid = PID(0.01, 0.00, 0.00, setpoint=0)  # TODO is setpoint=0 correct?
        # self.linear_speed = 0.05  # Assuming a default linear speed
        # self.linear_speed = 0.0
        self.linear_speed = 0.001

    def listener_callback(self, data):
        try:
            # twist_msg = Twist()
            # twist_msg.linear.x = 0.0
            # twist_msg.angular.z = 0.01  # Scale down to ensure it's not too high
            # # twist_msg.angular.z = correction * 0.5  # Scale down to ensure it's not too high
            # print('linear.x:', twist_msg.linear.x, ' angular z:',  twist_msg.angular.z)
            # self.cmd_vel_publisher_.publish(twist_msg)

            image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            self.image_width = w  # Set the image width for PID calculations

            # blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
            # blurred_gray = gray
            edges = cv2.Canny(blurred_gray, 30, 100, apertureSize=3)
            edges = cv2.dilate(edges, None, iterations=1)

            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            self.edges_publisher_.publish(edges_msg)

            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = image.copy()

            center_x = w // 2
            min_contour_points = 50  # Minimum number of points for a contour to be considered
            if contours:
                best_score = float('inf')
                chosen_contour = None  # Initialize with None to handle case where no contour meets criteria
                for contour in contours:
                    if len(contour) >= min_contour_points:  # Check if contour has enough points
                        rect = cv2.minAreaRect(contour)  # Get the minimum area rectangle for the contour
                        box = cv2.boxPoints(rect)  # Get the four points of the rectangle
                        box = np.int0(box)  # Convert points to integers
                        width = np.linalg.norm(box[0] - box[1])
                        height = np.linalg.norm(box[1] - box[2])
                        aspect_ratio = width / height if width > height else height / width
                        
                        # Check if the contour is line-like by its aspect ratio
                        if 2 < aspect_ratio < 10:  # Adjust the range as necessary
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                distance = abs(cX - center_x)
                                # Prioritize contours closer to the bottom of the image (higher y value means lower on the image)
                                y_priority = (h - cY) / h
                                # Combine distance, y_priority, and aspect ratio into a single score for comparison
                                # Adjust weighting as necessary. Here, we give more weight to the y position to prioritize lower contours
                                score = distance + (y_priority * 100) - (aspect_ratio * 10)  # Subtract aspect ratio to prioritize line-like contours
                                if score < best_score:
                                    best_score = score
                                    chosen_contour = contour
                if chosen_contour is None and contours:  # Fallback to the first contour if none meet the criteria
                    chosen_contour = contours[0]
            else:
                chosen_contour = None

            # # After finding contours
            # color_threshold = 150
            # for contour in contours:
            #     if len(contour) >= min_contour_points:  # Check if contour has enough points
            #         # Create a mask image that contains the contour filled in
            #         mask = np.zeros_like(gray)
            #         cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

            #         # Use the mask to calculate the mean color of the contour area in the original image
            #         mean_val = cv2.mean(image, mask=mask)

            #         # Check if the mean color is dark (i.e., on the black line)
            #         # Assuming the image is in BGR format and we're looking for a dark color
            #         # You may need to adjust the threshold value based on your specific image
            #         # if mean_val[0] < color_threshold and mean_val[1] < color_threshold and mean_val[2] < color_threshold:
            #         #     # This contour is on the black line, process it
            #         #     # ...

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
            correction, error, cX, cY = self.use_chosen_contour_for_pid_set_point(chosen_contour)

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

            # Update the processed image message with the direction information
            processed_image_msg = self.bridge.cv2_to_imgmsg(image_with_contours, encoding="passthrough")
            self.publisher_.publish(processed_image_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to process and publish image: {e}. Exception occurred at line: {e.__traceback__.tb_lineno}')
    
    
    def use_chosen_contour_for_pid_set_point(self, chosen_contour):
        if chosen_contour is not None:
            y_positions = chosen_contour[:, :, 1].flatten()
            highest_y_index = np.argmax(y_positions)
            highest_y_point = chosen_contour[highest_y_index][0]
            cX, cY = highest_y_point[0], highest_y_point[1]
            error = cX - self.image_width // 2
            raw_correction = self.pid(error)  # Changed from self.pid.update(error)
            # Normalize correction to be proportional to linear speed, ensuring it's within a suitable range for PID control
            # correction = np.tanh(raw_correction) * 0.05
            # correction = np.tanh(raw_correction) * 0.01
            correction = raw_correction * 0.02
            
            # correction = 0.0
            twist_msg = Twist()
            twist_msg.linear.x = self.linear_speed
            twist_msg.angular.z = correction  # Scale down to ensure it's not too high
            # twist_msg.angular.z = correction * 0.5  # Scale down to ensure it's not too high
            # twist_msg.linear.x = 0.0
            # twist_msg.angular.z = 0.0

            print('linear.x:', twist_msg.linear.x, ' angular z:',  correction)
            # self.cmd_vel_publisher_.publish(twist_msg)

            self.get_logger().info(f"Centroid: ({cX}, {cY}), Error: {error}, raw_correction: {raw_correction}, Correction: {correction}")
            return correction, error, cX, cY
            # else:
            #     self.get_logger().warn("Chosen contour has zero area, skipping PID control.")
            #     return None, None, None, None
        else:
            self.get_logger().warn("No contour chosen, skipping PID control.")
            return None, None, None, None
        

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()