import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, '/webcam/image/compressed', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                compressed_img_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
                self.publisher_.publish(compressed_img_msg)
            except Exception as e:
                self.get_logger().error('Failed to convert and publish: %r' % (e,))

def main(args=None):
    rclpy.init(args=args)
    webcam_publisher = WebcamPublisher()
    rclpy.spin(webcam_publisher)
    webcam_publisher.cap.release()
    webcam_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()