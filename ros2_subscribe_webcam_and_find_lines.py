import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2

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
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
            processed_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
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