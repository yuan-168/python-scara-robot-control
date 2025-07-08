import numpy as np
import math
# Importing necessary libraries from ROS2 Python client library
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Import the String message type from standard ROS2 message library
from std_msgs.msg import Float32MultiArray 

# Defining the Kinematics class which inherits from Node
class JacobianBasedContoller(Node):

    def __init__(self):
        super().__init__('jacobian_based_controller')  # Initialize the node with the name 'jacobian_based_controller'
        self.l1 = 10  #link length 1(cm)
        self.l2 = 10  #link length 2(cm)
        self.l3 = 10  #link length 3(cm)
        self.target_position = np.array([0,0])
        
        self.subscription = self.create_subscription(Float32MultiArray,'joint_state', self.joint_state_callback, 10)
        # Create a publisher object with Float message type on the topic 'joint_pos_rel'
        # The second argument '10' is the queue size
        self.publisher = self.create_publisher(Float32MultiArray, 'joint_pos_rel', 10)

    def joint_state_callback(self, msg):
        # Get data from publisher (since we are subscribing)
        data = msg.data # 3x angles, 3x vel, 3x current
        current_joint_angles = np.radians(data[0:3])
        
        # call FK using current angles to obtain current pos
        current_position = self.forward_kinematics(current_joint_angles)

        # set current position to initial position
        if self.target_position.all() == 0:
            self.target_position = current_position

        # compute delta_x = target_pos - current_position
        delta_X = self.target_position - current_position

        #if np.linalg.norm(delta_X) > 0.05:
            # compute the Jacobian to get delta_q
        jacobian = self.compute_analytic_jacobian(current_joint_angles)
        pseudo_inv_jacobian = np.linalg.pinv(jacobian)
        delta_Q = np.dot(pseudo_inv_jacobian,delta_X)

            # send data back to hw_interface (using publisher)
        return_msg = Float32MultiArray()
        return_msg.data = [np.rad2deg(delta_Q[0]),np.rad2deg(delta_Q[1]),np.rad2deg(delta_Q[2])]
        self.publisher.publish(return_msg)

    # Compute pose of end effector
    def forward_kinematics(self, angles):
        th1 = angles[0]
        th2 = angles[1]
        th3 = angles[2]
        x = self.l1*math.cos(th1) + self.l2*math.cos(th1+th2) + self.l3*math.cos(th1+ th2+ th3)
        y = self.l1*math.sin(th1) + self.l2*math.sin(th1+th2) + self.l3*math.sin(th1+ th2+ th3)
        return np.array([x,y])

    # Compute analytic jacobian
    def compute_analytic_jacobian(self,angles):
        # get angles th1,th2,th3
        th1 = angles[0]
        th2 = angles[1]
        th3 = angles[2]
        #compute jacobian matrix
        jacobian = np.array([[-self.l1*math.sin(th1)-self.l2*math.sin(th1+th2)-self.l3*math.sin(th1+th2+th3), -self.l2*math.sin(th1+th2)-self.l3*math.sin(th1+th2+th3), -self.l3*math.sin(th1+th2+th3)],
                              [self.l1*math.cos(th1)+self.l2*math.cos(th1+th2)+self.l3*math.cos(th1+th2+th3), self.l2*math.cos(th1+th2)+self.l3*math.cos(th1+th2+th3), self.l3*math.cos(th1+th2+th3)]
                              ])
        return jacobian
    

# The main function which serves as the entry point for the program
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 Python client library
    jacobian_based_controller = JacobianBasedContoller()  # Create an instance of Jacobian Based Controller

    try:
        rclpy.spin(jacobian_based_controller)  # Keep the node alive and listening for messages
    except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
        pass

    jacobian_based_controller.destroy_node()  # Properly destroy the node
    rclpy.shutdown()  # Shutdown the ROS2 Python client library

# This condition checks if the script is executed directly (not imported)
if __name__ == '__main__':
    main()  # Execute the main function
