import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyToTwist(Node):
    def __init__(self):
        super().__init__('joy_to_twist')
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

    def joy_callback(self, msg):
        twist = Twist()
        # Example mapping: left joystick vertical for linear velocity,
        # right joystick horizontal for angular velocity.
        twist.linear.x = msg.axes[1]
        twist.angular.z = msg.axes[3]
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    joy_to_twist = JoyToTwist()
    rclpy.spin(joy_to_twist)
    joy_to_twist.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()