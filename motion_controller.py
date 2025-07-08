import numpy as np
import math
# Importing necessary libraries from ROS2 Python client library
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Import the String message type from standard ROS2 message library
from std_msgs.msg import Float32MultiArray 

# Defining the MotionController class which inherits from Node
class MotionController(Node):

    def __init__(self):
        super().__init__('motion_controller')  # Initialize the node with the name 'motion_controller'

        # Link lengths in cm
        self.l1 = 10
        self.l2 = 10
        self.l3 = 10
        # Target position
        self.target_position = np.array([20,-10])
        # Step size for trajectory
        self.max_step = 2.5

        # Subscribe to topic 'joint_state'
        self.subscription = self.create_subscription(Float32MultiArray,'joint_state', self.joint_state_callback, 10)
        # Create a publisher object with Float message type on the topic 'joint_pos_rel'
        # The second argument '10' is the queue size
        self.publisher = self.create_publisher(Float32MultiArray, 'joint_pos_rel', 10)

    def joint_state_callback(self, msg):
        # get data from publisher (since we are subscribing)
        data = msg.data # 3x angles, 3x vel, 3x current
        th1 = np.radians(data[0])
        th2 = np.radians(data[1])
        th3 = np.radians(data[2])
        
        # Call FK using current angles to obtain current position
        current_position = self.forward_kinematics(th1,th2,th3)

        # Calculate magnitude and direction
        distance = np.linalg.norm(self.target_position - np.array(current_position))
        direction = (self.target_position - np.array(current_position)) / distance

        if distance >= self.max_step:
            delta_X = direction * self.max_step
        else:
            delta_X = self.target_position - np.array(current_position)

        # Compute the Jacobian to get delta_Q
        jacobian = self.compute_analytic_jacobian(th1,th2,th3)
        pseudo_inv_jacobian = np.linalg.pinv(jacobian)
        delta_Q = np.dot(pseudo_inv_jacobian,delta_X)
        delta_Q = np.rad2deg(delta_Q)

        # send data back to hw_interface (using publisher)
        return_msg = Float32MultiArray()
        return_msg.data = delta_Q.tolist()
        self.publisher.publish(return_msg)

    # Compute pose of end effector
    def forward_kinematics(self,th1,th2,th3):
        x = self.l1*math.cos(th1) + self.l2*math.cos(th1+th2) + self.l3*math.cos(th1+ th2+th3)
        y = self.l1*math.sin(th1) + self.l2*math.sin(th1+th2) + self.l3*math.sin(th1+ th2+th3)
        return x,y

    # Compute analytic jacobian
    def compute_analytic_jacobian(self,th1,th2,th3):
        jacobian = np.array([[-self.l1*math.sin(th1)-self.l2*math.sin(th1+th2)-self.l3*math.sin(th1+th2+th3), -self.l2*math.sin(th1+th2)-self.l3*math.sin(th1+th2+th3), -self.l3*math.sin(th1+th2+th3)],
                              [self.l1*math.cos(th1)+self.l2*math.cos(th1+th2)+self.l3*math.cos(th1+th2+th3), self.l2*math.cos(th1+th2)+self.l3*math.cos(th1+th2+th3), self.l3*math.cos(th1+th2+th3)]
                              ])
        return jacobian
    

# The main function which serves as the entry point for the program
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 Python client library
    motion_controller = MotionController()  # Create an instance of Kinematics

    try:
        rclpy.spin(motion_controller)  # Keep the node alive and listening for messages
    except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
        pass

    motion_controller.destroy_node()  # Properly destroy the node
    rclpy.shutdown()  # Shutdown the ROS2 Python client library

# This condition checks if the script is executed directly (not imported)
if __name__ == '__main__':
    main()  # Execute the main function
