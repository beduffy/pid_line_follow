import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from simple_pid import PID
from shapely.geometry import Polygon

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
        # TODO try visualise compressed image in rviz again
        self.publisher_ = self.create_publisher(Image, 'webcam/processed_image', 10)
        self.edges_publisher_ = self.create_publisher(Image, 'webcam/edges_image', 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.cmd_vel_publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.bridge = CvBridge()
        self.pid = PID(0.01, 0.00, 0.00, setpoint=0)  # TODO is setpoint=0 correct?
        # self.linear_speed = 0.05  # Assuming a default linear speed
        # self.linear_speed = 0.0
        self.linear_speed = 0.001


        self.linear_speed = 0.1
        self.linear_speed = 0.25
        self.linear_speed = 0.4

    def listener_callback(self, data):
        try:

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

            # old values for non tuned gucv
            # hueLow = 0
            # hueHigh = 179
            # satLow = 27
            # satHigh = 255
            # valLow = 200
            # valHigh = 226

            hueLow = 13
            hueHigh = 175
            satLow = 17
            satHigh = 196
            valLow = 130
            valHigh = 226

            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([hueLow, satLow, valLow])
            upper_yellow = np.array([hueHigh, satHigh, valHigh])
            center_x = w // 2
            image_third_height = h * 2 // 3
            min_contour_points = 5  # Minimum number of points for a contour to be considered
            chosen_contour = None
            max_yellow_nearby = 0  # Initialize maximum yellow nearby count
            yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

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


            # for contour in contours:
            #     if np.array_equal(contour, chosen_contour):
            #         cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 3)
            #         y_positions = contour[:, :, 1].flatten()
            #         lowest_y = np.min(y_positions)
            #         highest_y = np.max(y_positions)
            #         print(f"Lowest Y: {lowest_y}, Highest Y: {highest_y}")
            #     else:
            #         cv2.drawContours(image_with_contours, [contour], -1, (0, 0, 255), 3)

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
            import traceback
            tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            error_message = ''.join(tb_str)
            self.get_logger().error(f'Failed to process and publish image. Full error: {error_message}')
    
    def use_chosen_contour_for_pid_set_point(self, chosen_contour):
        if chosen_contour is not None:
            y_positions = chosen_contour[:, :, 1].flatten()
            sorted_indices = np.argsort(y_positions)
            one_third_index = sorted_indices[len(sorted_indices) // 3]
            one_third_point = chosen_contour[one_third_index][0]
            cX, cY = one_third_point[0], one_third_point[1]
            error = cX - self.image_width // 2
            raw_correction = self.pid(error)
            # Normalize correction to be proportional to linear speed, ensuring it's within a suitable range for PID control
            # correction = np.tanh(raw_correction) * 0.05
            # correction = raw_correction * 0.1  # worked well for 0.1m/s
            correction = raw_correction * 0.4  # worked well for 0.25
            correction = raw_correction * 0.8  # worked well for 0.4
            
            # TODO to go 0.54m/s I nened to turn off safety override
            twist_msg = Twist()
            twist_msg.linear.x = self.linear_speed
            twist_msg.angular.z = correction  # Scale down to ensure it's not too high

            print('linear.x:', twist_msg.linear.x, ' angular z:',  correction)
            self.cmd_vel_publisher_.publish(twist_msg)

            self.get_logger().info(f"Centroid: ({cX}, {cY}), Error: {error}, raw_correction: {raw_correction}, Correction: {correction}")
            return correction, error, cX, cY
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