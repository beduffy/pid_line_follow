import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import os
import datetime

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/webcam/image/compressed',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.image_folder = 'data/march_27'
        os.makedirs(self.image_folder, exist_ok=True)
        self.last_saved_time = None
        self.save_interval = 1.0 / 5  # 5 fps, so 1/5th of a second between saves

    def listener_callback(self, data):
        current_time = datetime.datetime.now()
        if self.last_saved_time is None or (current_time - self.last_saved_time).total_seconds() >= self.save_interval:
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
                timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S-%f")
                image_path = os.path.join(self.image_folder, f"{timestamp}.jpg")
                cv2.imwrite(image_path, cv_image)
                self.get_logger().info(f'Saved image to {image_path}')
                self.last_saved_time = current_time
            except Exception as e:
                self.get_logger().error(f'Failed to save image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
