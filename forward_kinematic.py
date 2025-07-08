import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import numpy as np



class Kinematics(Node):
   
    def __init__(self):
        super().__init__('kinematics')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/joint_state',
            self.joint_state_callback,
            10
        )
        self.subscription
        # Hold a reference to the subscription to prevent it from being garbage-collected
        # 创建用于关节位置发布的发布者
        # self.publisher = self.create_publisher(Float32MultiArray, 'joint_pos_rel', 10)
        self.publisher = self.create_publisher(Float32MultiArray, 'joint_cur', 10)
        # 初始化关节角度
        self.joint_angles = np.zeros(3)  # 3 是关节的数量
        #link lengths in mm
        self.r1 = 0.1
        self.r2 = 0.1
        self.r3 = 0.1
        self.target = np.array([0.3,0.0])
        self.max_step = 0.05
        
         # 初始化关节电流
        self.friction_coefficient = 1.06  # 摩擦力系数
        self.friction_offset = 10.3  # 摩擦力偏移

        # added by Jiaheng
        self.Kpvd1 = 5.0
        self.Kdvd1 = 1
        self.Kivd1 = 0.0
        self.Kpvd2 = 5.0
        self.Kdvd2 = 1
        self.Kivd2 = 0.0    
        self.Kpvd3 = 3.0
        self.Kdvd3 = 1
        self.Kivd3 = 0.0

        self.Pre_vel_err1 = 0 # 储存前项
        self.Int_vel_err1 = 0 # 储存积分
        self.Pre_vel_err2 = 0 # 储存前项
        self.Int_vel_err2 = 0 # 储存积分
        self.Pre_vel_err3 = 0 # 储存前项
        self.Int_vel_err3 = 0 # 储存积分        
        self.abs_int_limit = 10 # 储存积分上界
        self.k_cur_torque = 0.1 # T = K*I
        self.abs_u_limit = 50

    def joint_state_callback(self, joint_pos_msg):
       
       # 从接收到的JointState消息中提取关节角度
        joint_angles = joint_pos_msg.data
        th1 = np.radians(joint_angles[0])
        th2 = np.radians(joint_angles[1])
        th3 = np.radians(joint_angles[2])

        vel1 = joint_angles[3]
        vel2 = joint_angles[4]
        vel3 = joint_angles[5]

        # 调用正向运动学以获取当前位置
        current_position = self.forward_kinematics(th1,th2,th3)

        self.get_logger().info('Current position: "%s"' % np.array(current_position))
        
        # 计算当前位置与目标位置之间的差值
        # target_position = np.array([0.2, 0.1])  # 定义你的目标位置
        distance = np.linalg.norm(self.target - np.array(current_position))
        direction = (self.target - np.array(current_position)) / distance
        if abs(distance) >= self.max_step:
            delta_X = direction * self.max_step
        # elif abs(distance) < 0.0:
        #     delta_X = np.array([0,0])
        else:
            delta_X = self.target - np.array(current_position)
        # delta_X = self.target - np.array(current_position)
        
        # 雅可比矩阵函数
        J = self.jacobian([th1, th2, th3])
        
        # 使用雅可比矩阵的伪逆来计算关节角度的变化
        J_inv = np.linalg.pinv(J)
        # delta_angles = np.dot(J_inv, delta_X)
        delta_angles = J_inv @ delta_X
        
        # 更新关节角度变化
        delta_angles = np.rad2deg(delta_angles)  # 将弧度转换为度
        dqd1 = delta_angles[0]
        dqd2 = delta_angles[1]
        dqd3 = delta_angles[2]

        dqd1_err = dqd1 - vel1  
        dqd2_err = dqd2 - vel2  
        dqd3_err = dqd3 - vel3  
        
        self.Int_vel_err1 += dqd1_err
        self.Int_vel_err1 = max(-self.abs_int_limit, min(self.Int_vel_err1, self.abs_int_limit))
        self.Int_vel_err2 += dqd2_err
        self.Int_vel_err2 = max(-self.abs_int_limit, min(self.Int_vel_err2, self.abs_int_limit))
        self.Int_vel_err3 += dqd3_err
        self.Int_vel_err3 = max(-self.abs_int_limit, min(self.Int_vel_err3, self.abs_int_limit))

        ddqd1 = dqd1_err - self.Pre_vel_err1
        ddqd2 = dqd2_err - self.Pre_vel_err2
        ddqd3 = dqd3_err - self.Pre_vel_err3

        self.Pre_vel_err1 = dqd1_err
        self.Pre_vel_err2 = dqd2_err
        self.Pre_vel_err3 = dqd3_err

        u1 = self.pid_vel_control(self.Kpvd1, self.Kivd1, self.Kdvd1,dqd1_err,self.Int_vel_err1,ddqd1)
        u2 = self.pid_vel_control(self.Kpvd2, self.Kivd2, self.Kdvd2,dqd2_err,self.Int_vel_err2,ddqd2)
        u3 = self.pid_vel_control(self.Kpvd3, self.Kivd3, self.Kdvd3,dqd3_err,self.Int_vel_err3,ddqd3)

        u1_out = u1 + self.apply_friction_compensation(vel1,u1)
        u2_out = u2 + self.apply_friction_compensation(vel2,u2)
        u3_out = u3 + self.apply_friction_compensation(vel3,u3)

        # 创建一个新的消息用于关节角度变化
        joint_pos_rel_msg = Float32MultiArray()
        # joint_pos_rel_msg.data = delta_angles.tolist()  # 转换为列表以便兼容
        joint_pos_rel_msg.data = [u1_out,u2_out,u3_out]  # 转换为列表以便兼容

        # 发布关节角度变化
        self.publisher.publish(joint_pos_rel_msg)
        # Logging the published message to the console
        self.get_logger().info('Publishing: "%s"' % joint_pos_rel_msg.data)

        
    def pid_vel_control(self,kp,ki,kd,p,i,d):
        u = kp * p + ki * i + kd * d
        uout = max(-self.abs_u_limit, min(self.abs_u_limit,u)) 
        return uout
    
    def forward_kinematics(self, th1, th2, th3):
        px = self.r1 * np.cos(th1) + self.r2 * np.cos(th1 + th2) + self.r3 * np.cos(th1 + th2 + th3)
        py = self.r1 * np.sin(th1) + self.r2 * np.sin(th1 + th2) + self.r3 * np.sin(th1 + th2 + th3)
        
        return px, py

    def apply_friction_compensation(self, angle_vel, current):
        # 计算摩擦力补偿
        if current >= 0:
            return self.friction_coefficient * angle_vel + self.friction_offset
        else:
            return self.friction_coefficient * angle_vel - self.friction_offset
        
    def jacobian(self, th1, th2, th3):
        J = np.array([
        [-self.r1 * np.sin(th1) - self.r2 * np.sin(th1 + th2) - self.r3 * np.sin(th1 + th2 + th3),
         -self.r2 * np.sin(th1 + th2) - self.r3 * np.sin(th1 + th2 + th3),
         -self.r3 * np.sin(th1 + th2 + th3)],
        [self.r1 * np.cos(th1) + self.r2 * np.cos(th1 + th2) + self.r3 * np.cos(th1 + th2 + th3),
         self.r2 * np.cos(th1 + th2) - self.r3 * np.cos(th1 + th2 + th3),
         -self.r3 * np.cos(th1 + th2 + th3)]
        ])
        return J
    
    def jacobian(self,thetas):
        J = [[-self.r1*np.sin(thetas[0])-self.r2*np.sin(thetas[0]+thetas[1])-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2]),-self.r2*np.sin(thetas[0]+thetas[1])-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2]),-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2])],
                                [self.r1*np.cos(thetas[0])+self.r2*np.cos(thetas[0]+thetas[1])+self.r3*np.cos(thetas[0]+thetas[1]+thetas[2]),self.r2*np.cos(thetas[0]+thetas[1])+self.r3*np.cos(thetas[0]+thetas[1]+thetas[2]),self.r3*np.cos(thetas[0]+thetas[1]+thetas[2])]]
        return np.array(J)


# The main function which serves as the entry point for the program
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 Python client library
    kinematics_node = Kinematics() # Create an instance of the SimplePublisher

    try:
        rclpy.spin(kinematics_node)  # Keep the node alive and listening for messages
    except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
        pass

    kinematics_node.destroy_node()  # Properly destroy the node
    rclpy.shutdown()  # Shutdown the ROS2 Python client library
   

# This condition checks if the script is executed directly (not imported)
if __name__ == '__main__':
    main()  # Execute the main function
