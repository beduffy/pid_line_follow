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
            min_contour_points = 5  # Minimum number of points for a contour to be considered
            if contours:
                best_score = float('inf')
                chosen_contour_pair = None  # Initialize with None to handle case where no contour meets criteria
                parallel_contours = []  # Store pairs of parallel contours
                for i, contour1 in enumerate(contours):
                    if len(contour1) < min_contour_points:
                        print('continuing', len(contour1))
                        continue
                    for j, contour2 in enumerate(contours[i+1:], start=i+1):  # Avoid comparing the same contours
                        if len(contour2) < min_contour_points:
                            print('continuing, contour points', len(contour2))
                            continue
                        # Calculate the bounding rotated rectangles for each contour
                        rect1 = cv2.minAreaRect(contour1)
                        box1 = cv2.boxPoints(rect1)
                        box1 = np.int0(box1)

                        rect2 = cv2.minAreaRect(contour2)
                        # Calculate the angle of each rectangle
                        angle1 = rect1[-1]
                        angle2 = rect2[-1]
                        # Normalize angles to the range [0, 180)
                        if angle1 < -45:
                            angle1 = 90 + angle1
                        if angle2 < -45:
                            angle2 = 90 + angle2
                        angle1 = abs(angle1)
                        angle2 = abs(angle2)
                        # Calculate the angle difference
                        angle_diff = abs(angle1 - angle2)
                        # Display angle difference on the image near the contour for debugging
                        box2 = cv2.boxPoints(rect2)
                        box2 = np.int0(box2)
                        cv2.drawContours(image_with_contours, [box1], 0, (0, 255, 0), 2)
                        cv2.drawContours(image_with_contours, [box2], 0, (0, 0, 255), 2)
                        # Calculate the center of the second rectangle to place the text
                        M2 = cv2.moments(contour2)
                        if M2["m00"] != 0:
                            cX2 = int(M2["m10"] / M2["m00"])
                            cY2 = int(M2["m01"] / M2["m00"])
                            cv2.putText(image_with_contours, f"Diff: {angle_diff:.2f}", (cX2, cY2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # Check if the contours are roughly parallel by comparing their angles
                        if angle_diff < 10 or angle_diff > 350:  # Allowing a small angle difference
                            # Calculate the center points of each contour
                            M1 = cv2.moments(contour1)
                            M2 = cv2.moments(contour2)
                            # print(M1, M2)
                            if M1["m00"] != 0 and M2["m00"] != 0:
                                cX1 = int(M1["m10"] / M1["m00"])
                                cY1 = int(M1["m01"] / M1["m00"])
                                cX2 = int(M2["m10"] / M2["m00"])
                                cY2 = int(M2["m01"] / M2["m00"])
                                # Calculate the distance between the center points of the contours
                                center_distance = np.sqrt((cX2 - cX1)**2 + (cY2 - cY1)**2)
                                # print(f'Center distance between contours: {center_distance}')
                                # Check if the distance is within a reasonable range to consider them as a pair
                                if 1 < center_distance < 100:  # Adjust the range as necessary
                                    print('appending, center distance: ', center_distance)
                                    parallel_contours.append((contour1, contour2))
                # Choose the best pair based on their proximity to the center and their combined area
                best_score = float('inf')
                chosen_contour_pair = None
                for contour_pair in parallel_contours:
                    contour1, contour2 = contour_pair
                    M1 = cv2.moments(contour1)
                    M2 = cv2.moments(contour2)
                    cX1 = int(M1["m10"] / M1["m00"])
                    cX2 = int(M2["m10"] / M2["m00"])
                    # Calculate the average x position of the pair
                    avg_cX = (cX1 + cX2) / 2
                    distance = abs(avg_cX - center_x)
                    # Calculate the combined area of the contours
                    area1 = cv2.contourArea(contour1)
                    area2 = cv2.contourArea(contour2)
                    combined_area = area1 + area2
                    # Score based on distance to center and combined area (prioritize closer and larger pairs)
                    score = distance - combined_area
                    if score < best_score:
                        best_score = score
                        chosen_contour_pair = contour_pair  # Update chosen_contour_pair to be the best pair
            # chosen_contour = chosen_contour_pair  # Assign the chosen pair to chosen_contour for further processing
            chosen_contour = chosen_contour_pair[0] if chosen_contour_pair else None  # Assign the chosen pair to chosen_contour for further processing

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
            correction = raw_correction * 0.02
            
            twist_msg = Twist()
            twist_msg.linear.x = self.linear_speed
            twist_msg.angular.z = correction  # Scale down to ensure it's not too high

            print('linear.x:', twist_msg.linear.x, ' angular z:',  correction)
            # self.cmd_vel_publisher_.publish(twist_msg)

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