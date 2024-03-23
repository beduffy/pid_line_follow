import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2


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
        # self.publisher_ = self.create_publisher(CompressedImage, '/processed/image/compressed', 10)
        self.publisher_ = self.create_publisher(Image, 'webcam/processed_image', 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        try:
            # Convert ROS CompressedImage to OpenCV image
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
            # Process the image (convert to grayscale in this example)
            # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Convert back to ROS CompressedImage and publish
            # processed_img_msg = self.bridge.cv2_to_compressed_imgmsg(gray_image)
            image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
            
            #############################
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

                # if contour_is_in_bottom_half and contour.shape[0] > 300:
                print('Num contours: ', contour.shape[0])
                # image_with_contours = image_with_contours.copy()
                
                # contour_is_line_like = is_line_like(contour)
                # print('contour_is_line_like:', contour_is_line_like)
                
                cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)



            #############################
            
            
            processed_image_msg = self.bridge.cv2_to_imgmsg(image_with_contours, encoding="passthrough")
            self.publisher_.publish(processed_image_msg)
        except Exception as e:
            self.get_logger().error('Failed to process and publish image: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()