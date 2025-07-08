import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import numpy as np
import csv
import time


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

        self.csv_file = open('joint_angles_and_currents.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time', 'Joint Angle 1', 'Joint Angle 2', 'Joint Angle 3', 'Current 1', 'Current 2', 'Current 3'])
        # 初始化关节角度
        self.joint_angles = np.zeros(3)  # 3 是关节的数量
        #link lengths in mm
        self.r1 = 0.1
        self.r2 = 0.1
        self.r3 = 0.1
        self.circle_center = np.array([0.15,0.0])
        self.circle_r = 0.05
        self.x_min = 0.25
        self.abs_y_max = 0.075
        self.force_magnitude = 1
        self.max_magnitude = 4
        # self.aim_angles, _ = self.inverse_kinematics
        self.max_step = 0.05

         # 初始化关节电流
        self.friction_coefficient = 1.06  # 摩擦力系数
        self.friction_offset = 18.5  # 摩擦力偏移
        self.friction_offset2 = 15.0  # 摩擦力偏移
        self.friction_offset3 = 13.0  # 摩擦力偏移
        self.smooth_range = 0.04

        # added by Jiaheng
        self.Kpvd1 = 5.0
        self.Kdvd1 = 0.50
        self.Kivd1 = 0.0
        self.Kpvd2 = 5.0
        self.Kdvd2 = 0.50
        self.Kivd2 = 0.0
        self.Kpvd3 = 100.0
        self.Kdvd3 = 0.50
        self.Kivd3 = 0.0

        self.Pre_pos_err1 = 0
        self.Pre_vel_err1 = 0 # 储存前项
        self.Pre2_vel_err1 = 0 #
        self.Int_vel_err1 = 0 # 储存积分
        self.Pre_pos_err2 = 0
        self.Pre_vel_err2 = 0 # 储存前项
        self.Pre2_vel_err2 = 0 #
        self.Int_vel_err2 = 0 # 储存积分
        self.Pre_pos_err3 = 0
        self.Pre_vel_err3 = 0 # 储存前项
        self.Pre2_vel_err3 = 0 #
        self.Int_vel_err3 = 0 # 储存积分
        self.abs_int_limit = 5000 # 储存积分上界
        self.k_cur_torque = 0.278 * 0.001 # T = k*I + b
        self.b_cur_torque = 0.112
        self.abs_u_limit = 50
        self.lambda_damping = 0.01
        self.inverse_kine_rate = 1
        self.inverse_kine_disRange = 0.001
        self.inverse_kine_maxIter = 5

        # added by Jiaheng, for dynamic control
        self.m_joint = 18 * 0.001
        self.m_arm = 27.83 * 0.001
        self.center_of_mass = 0.1 * 1
        self.m_o_i = 0.000054

    def calc_T_from_I(self,cur):
        return (cur * self.k_cur_torque + np.sign(cur) * self.b_cur_torque)

    def calc_I_from_T(self,tor):
        return (tor / self.k_cur_torque)

    def joint_state_callback(self, joint_pos_msg):

       # 从接收到的JointState消息中提取关节角度
        joint_angles = joint_pos_msg.data
        th1 = np.radians(joint_angles[0])
        th2 = np.radians(joint_angles[1])
        th3 = np.radians(joint_angles[2])

        vel1 = np.radians(joint_angles[3])
        vel2 = np.radians(joint_angles[4])
        vel3 = np.radians(joint_angles[5])

        current_time = time.time()
        self.csv_writer.writerow([current_time] + list(joint_angles) + list(currents))

    def __del__(self):
        self.csv_file.close()


        # 调用正向运动学以获取当前位置
        current_position = self.forward_kinematics(th1,th2,th3)

        vector_to_center = self.circle_center - current_position
        distance_to_center = np.linalg.norm(vector_to_center)
        force = np.array([0.0,0.0])
        # if distance_to_center > self.circle_r:
        #     # Normalize the vector to center to get the direction
        #     direction_to_center = vector_to_center / distance_to_center
        #     # Calculate the force components
        #     force = direction_to_center * self.force_magnitude * (distance_to_center - self.circle_r) / self.circle_r
        # else:
        #     force = np.array([0.0,0.0])
        if current_position[0] < self.x_min:
            force[0] = min((self.x_min - current_position[0])/self.x_min,self.max_magnitude) * self.force_magnitude
            # force[0] = self.force_magnitude

        if current_position[1] > self.abs_y_max:
            force[1] = -1 * min((current_position[1] - self.abs_y_max)/self.abs_y_max,self.max_magnitude) * self.force_magnitude
            # force[1] = -self.force_magnitude
        if current_position[1] < -self.abs_y_max:
            force[1] = min((-self.abs_y_max - current_position[1])/self.abs_y_max,self.max_magnitude) * self.force_magnitude
            # force[1] = self.force_magnitude

        # 雅可比矩阵函数
        J = self.jacobian([th1, th2, -th1-th2])

        # 使用雅可比矩阵的伪逆来计算关节角度的变化
        J_inv = np.linalg.pinv(J)
        # I = np.eye(J.shape[0])
        torques = J.T @ force

        m = self.get_M_matrix_new([th1,th2,th3])
        c = self.get_C_matrix([th1,th2,th3],[vel1,vel2,vel3])

        pd3 = -th1-th2 - th3
        dpd3 = 0
        ipd3 = 0
        u3 = self.pid_vel_control(self.Kpvd3,self.Kivd3,self.Kdvd3,pd3,ipd3,dpd3)

        # u1_out = u1 + self.apply_friction_compensation(vel1,u1)
        # u2_out = u2 + self.apply_friction_compensation(vel2,u2)
        # u3_out = u3 + self.apply_friction_compensation(vel3,u3)
        u1_out = self.calc_I_from_T(torques[0]) + self.apply_friction_compensation(vel1,vel1,self.friction_offset)
        u2_out = self.calc_I_from_T(torques[1]) + self.apply_friction_compensation(vel2,vel2,self.friction_offset2)
        # u3_out = self.calc_I_from_T(torques[2]) + self.apply_friction_compensation(vel3,vel3,self.friction_offset3)
        # u3_out = u3 + self.apply_friction_compensation(vel3,vel3,self.friction_offset3)
        u3_out = self.calc_I_from_T(torques[2]) + u3 + self.apply_friction_compensation(vel3,vel3,self.friction_offset3)

        # u1_out = self.calc_I_from_T(T[0])*0 + self.apply_friction_compensation(vel1,vel1,self.friction_offset3)
        # u2_out = self.calc_I_from_T(T[1])*0 + self.apply_friction_compensation(vel1,vel2,self.friction_offset2)
        # u3_out = self.calc_I_from_T(T[2])*0 + self.apply_friction_compensation(vel1,vel3,self.friction_offset)
        # 创建一个新的消息用于关节角度变化
        joint_pos_rel_msg = Float32MultiArray()
        # joint_pos_rel_msg.data = delta_angles.tolist()  # 转换为列表以便兼容
        joint_pos_rel_msg.data = [u1_out,u2_out,u3_out]  # 转换为列表以便兼容

        # 发布关节角度变化
        self.publisher.publish(joint_pos_rel_msg)
        # Logging the published message to the console
        self.get_logger().info('Publishing: "%s"' % joint_pos_rel_msg.data)
        # self.get_logger().info('Current position: {}\n aim: {}\n aim_angles: {}\n err : {}\n T : {}\n fric : {}\n input{}'.format(current_position,self.forward_kinematics(1 * delta_angles[0] + th1, 1 * delta_angles[1] + th2, 1 * delta_angles[2] + th3),self.aim_angles,delta_angles,T,self.debug_apply_friction_compensation(vel2,self.calc_I_from_T(T[0]),self.friction_offset2),[u1_out,u2_out,u3_out]))
    def save_data_to_csv(self, th1, th2, th3,):
        with open("force_transmission.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "th1", "th2", "Force Transmission"])
            writer.writerows(self.force_transmissions)

    def get_M_matrix_new(self, thetas):
        q1, q2, q3 = thetas
        m = self.m_joint + self.m_arm
        c2_3 = np.cos(q2 + q3)
        c2 = np.cos(q2)
        c3 = np.cos(q3)
        s2_3 = np.sin(q2 + q3)
        s2 = np.sin(q2)
        s3 = np.sin(q3)

        m11 = 2*self.center_of_mass**2*m + self.center_of_mass**2*self.m_arm + 3*m*self.r1**2 + 3*self.m_arm*self.r1**2 + 2*self.m_arm*self.r1**2*c2_3 + 2*m*self.r1**2*c2 + 2*self.m_arm*self.r1**2*c2 + 2*self.m_arm*self.r1**2*c3 + 4*self.center_of_mass*m*self.r1 + 2*self.center_of_mass*self.m_arm*self.r1 + 2*self.center_of_mass*self.m_arm*self.r1*c2_3 + 2*self.center_of_mass*m*self.r1*c2 + 2*self.center_of_mass*self.m_arm*self.r1*c3
        m12 = self.m_arm*(self.center_of_mass + self.r1)*(self.center_of_mass + self.r1 + self.r1*c2_3 + self.r1*c3) + m*(self.center_of_mass + self.r1)*(self.center_of_mass + self.r1 + self.r1*c2) + self.m_arm*self.r1*c3*(self.center_of_mass + self.r1 + self.r1*c2_3 + self.r1*c3) + self.m_arm*self.r1**2*s3*(s2_3 + s3)
        m13 = self.m_arm*(self.center_of_mass + self.r1)*(self.center_of_mass + self.r1 + self.r1*c2_3 + self.r1*c3)

        m21 = m12
        m22 = self.center_of_mass**2*m + self.center_of_mass**2*self.m_arm + m*self.r1**2 + 2*self.m_arm*self.r1**2 + 2*self.m_arm*self.r1**2*c3 + 2*self.center_of_mass*m*self.r1 + 2*self.center_of_mass*self.m_arm*self.r1 + 2*self.center_of_mass*self.m_arm*self.r1*c3
        m23 = self.m_arm*(self.center_of_mass + self.r1)**2 + self.m_arm*self.r1*c3*(self.center_of_mass + self.r1)


        m31 = m13
        m32 = m23
        m33 = self.m_arm * (self.center_of_mass + self.r1) ** 2

        return np.array([[m11, m12, m13],
                         [m21, m22, m23],
                         [m31, m32, m33]])

    def get_M_matrix(self,thetas):

        # Constants using provided formulas
        q1, q2, q3 = thetas
        m = self.m_joint + self.m_arm
        lc = self.center_of_mass
        self.r1 = self.r1
        J = self.m_o_i

        c1 = m * lc**2 + m * self.r1**2 + J
        c2 = m * lc**2 + J
        c3 = m * self.r1 * lc
        c6 = J + m * (lc**2 + self.r1**2 + self.r1**2)
        c7 = m * self.r1 * lc
        c8 = m * self.r1 * lc
        c9 = m * self.r1 * self.r1
        c10 = J + m * (lc**2 + self.r1**2)
        c11 = J + m * lc**2

        d11 = c1 + c2 + c6 + 2 * (c3 + c9) * np.cos(q2) + 2 * c7 * np.cos(q2 + q3) + 2 * c8 * np.cos(q3)
        d12 = c2 + c10 + (c3 + c9) * np.cos(q2) + c7 * np.cos(q2 + q3) + 2 * c8 * np.cos(q3)
        d13 = c11 + c7 * np.cos(q2 + q3) + c8 * np.cos(q3)

        d21 = d12
        d22 = c2 + c10 + 2 * c8 * np.cos(q3)
        d23 = c11 + c8 * np.cos(q3)

        d31 = d13
        d32 = d23
        d33 = c11

        # Construct the inertia matrix M without the fourth row and column
        M = np.array([
            [d11, d12, d13],
            [d21, d22, d23],
            [d31, d32, d33]
        ])
        return M

    def get_C_matrix_new(self,thetas,dthetas):
        # Unpack the joint velocities
        q1, q2, q3 = thetas
        dq1, dq2, dq3 = dthetas
        m = self.m_joint + self.m_arm
        c2 = np.cos(q2)
        c3 = np.cos(q3)
        s2 = np.sin(q2)
        s3 = np.sin(q3)


        m11 = -dq2 * self.r1 * m * (self.center_of_mass * np.sin(q2 + q3) + self.r1 * np.sin(q2 + q3) + self.center_of_mass * np.sin(q2) + 2 * self.r1 * np.sin(q2)) - dq3 * self.r1 * m * (self.center_of_mass + self.r1) * (np.sin(q2 + q3) + np.sin(q3))
        m12 = -self.r1 * m * (self.center_of_mass * dq1 * np.sin(q2) + self.center_of_mass * dq2 * np.sin(q2) + self.center_of_mass * dq3 * np.sin(q3) + 2 * dq1 * self.r1 * np.sin(q2) + 2 * dq2 * self.r1 * np.sin(q2) + dq3 * self.r1 * np.sin(q3) + self.center_of_mass * dq1 * np.sin(q2 + q3) + self.center_of_mass * dq2 * np.sin(q2 + q3) + self.center_of_mass * dq3 * np.sin(q2 + q3) + dq1 * self.r1 * np.sin(q2 + q3) + dq2 * self.r1 * np.sin(q2 + q3) + dq3 * self.r1 * np.sin(q2 + q3))
        m13 = -self.r1 * m * (self.center_of_mass + self.r1) * (np.sin(q2 + q3) + np.sin(q3)) * (dq1 + dq2 + dq3)

        m21 = self.r1 * m * (self.center_of_mass * dq1 * np.sin(q2) - self.center_of_mass * dq3 * np.sin(q3) + 2 * dq1 * self.r1 * np.sin(q2) - dq3 * self.r1 * np.sin(q3) + self.center_of_mass * dq1 * np.sin(q2 + q3) + dq1 * self.r1 * np.sin(q2 + q3))
        m22 = -dq3 * self.r1 * m * np.sin(q3) * (self.center_of_mass + self.r1)
        m23 = -self.r1 * m * np.sin(q3) * (self.center_of_mass + self.r1) * (dq1 + dq2 + dq3)

        m31 = self.r1 * m * (self.center_of_mass + self.r1) * (dq1 * np.sin(q2 + q3) + dq1 * np.sin(q3) + dq2 * np.sin(q3))
        m32 = self.r1 * m * np.sin(q3) * (dq1 + dq2) * (self.center_of_mass + self.r1)
        m33 = 0

        return np.array([[m11, m12, m13],
                         [m21, m22, m23],
                         [m31, m32, m33]])

    def get_C_matrix(self,thetas,dthetas):
        # Unpack the joint velocities
        q1, q2, q3 = thetas
        dq1, dq2, dq3 = dthetas

        # Constants using provided formulas
        m = self.m_joint + self.m_arm
        lc = self.center_of_mass
        self.r1 = self.r1
        J = self.m_o_i
        c3 = m * self.r1 * lc
        c7 = m * self.r1 * lc
        c8 = m * self.r1 * lc
        c9 = m * self.r1 * self.r1

        # Coriolis and centrifugal terms (simplified as Beta_c3 is zero)
        # h1 = -c3
        # h2 = -c9
        # h3 = -c7
        # h4 = -c8
        # Coriolis and centrifugal terms with Beta_c3 set to zero
        h1 = -c3 * np.sin(q2)
        h2 = -c9 * np.sin(q2)
        h3 = -c7 * np.sin(q2 + q3)
        h4 = -c8 * np.sin(q3)
        h = h1 + h2 + h3
        j = h3 + h4

        CQQ11 = h * dq2 + j * dq3
        CQQ12 = h * (dq1 + dq2) + j * dq3
        CQQ13 = j * (dq1 + dq2 + dq3)

        CQQ21 = -h * dq1 + h4 * dq3
        CQQ22 = h4 * dq3
        CQQ23 = h4 * (dq1 + dq2 + dq3)

        CQQ31 = -j * dq1 - h4 * dq2
        CQQ32 = -h4 * (dq1 + dq2)
        CQQ33 = 0

        # # Compute Coriolis forces for each joint
        # CQQ1 = CQQ11 * dq1 + CQQ12 * dq2 + CQQ13 * dq3
        # CQQ2 = CQQ21 * dq1 + CQQ22 * dq2 + CQQ23 * dq3
        # CQQ3 = CQQ31 * dq1 + CQQ32 * dq2 + CQQ33 * dq3

        # # Construct the Coriolis matrix C without the fourth row and column
        # C = np.array([CQQ1, CQQ2, CQQ3])
        C = np.array([
            [CQQ11, CQQ12, CQQ13],
            [CQQ21, CQQ22, CQQ23],
            [CQQ31, CQQ32, CQQ33]
        ])

        return C

    def pid_vel_control(self,kp,ki,kd,p,i,d):

        u = kp * p + ki * i + kd * d
        uout = max(-self.abs_u_limit, min(self.abs_u_limit,u))

        return uout

    def dynamic_vel_control(self,kp,ki,kd,p,i,d,dd,m,c):
        U = kp * np.array(p) + kd * np.array(d) + np.array(dd) + ki * np.array(i)
        # U = np.dot(kp,)
        T = np.dot(m,U) + np.dot(c,np.array(d))
        # T = np.dot(c,np.array(d))
        # T = np.dot(m,U)
        return T

    def forward_kinematics(self, th1, th2, th3):
        px = self.r1 * np.cos(th1) + self.r2 * np.cos(th1 + th2) + self.r3 * np.cos(th1 + th2 + th3)
        py = self.r1 * np.sin(th1) + self.r2 * np.sin(th1 + th2) + self.r3 * np.sin(th1 + th2 + th3)

        return np.array([px, py])
        # return px, py

    def inverse_kinematics(self,start_thetas,aim_pos):
        theta = np.array(start_thetas)
        for i in range(self.inverse_kine_maxIter):
            # Compute the current end-effector position
            current_x = self.forward_kinematics(theta[0],theta[1],theta[2])

            # Compute the error between the desired and current positions
            e = aim_pos - current_x

            # Check if the error norm is within the acceptable threshold
            if np.linalg.norm(e) < self.inverse_kine_disRange:
                return theta, True

            # Compute the update for theta using the transpose of the Jacobian
            delta_theta = self.inverse_kine_rate * self.sudo_inverse_jaco(theta) @ e

            # Update theta
            theta += delta_theta

        # If we reach here, the maximum number of iterations was reached without convergence
        return theta, False

    def apply_friction_compensation(self, angle_vel, current, friction_offset = 10.3):
        # 计算摩擦力补偿
        if -self.smooth_range <= current <= self.smooth_range:
            # Linear interpolation within the smooth range
            smooth_friction_offset = friction_offset * (abs(current) + 0 * self.smooth_range) / (1 * self.smooth_range)
        else:
            # Outside the smooth range, use the full friction offset
            smooth_friction_offset = friction_offset
        return self.friction_coefficient * angle_vel + np.sign(current) * smooth_friction_offset

        # 计算摩擦力补偿
        # if current >= 0:
        #     return self.friction_coefficient * angle_vel + friction_offset
        # else:
        #     return self.friction_coefficient * angle_vel - friction_offset
        # Calculate friction compensation

    def debug_apply_friction_compensation(self, angle_vel, current, friction_offset = 10.3):
        if -self.smooth_range <= current <= self.smooth_range:
                    # Linear interpolation within the smooth range
            smooth_friction_offset = friction_offset * (abs(current) + self.smooth_range) / (2 * self.smooth_range)
        else:
            # Outside the smooth range, use the full friction offset
            smooth_friction_offset = friction_offset

        # Calculate friction compensation
        return np.sign(current)

    def jacobian(self,thetas):
        J = [[-self.r1*np.sin(thetas[0])-self.r2*np.sin(thetas[0]+thetas[1])-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2]),-self.r2*np.sin(thetas[0]+thetas[1])-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2]),-self.r3*np.sin(thetas[0]+thetas[1]+thetas[2])],
                                [self.r1*np.cos(thetas[0])+self.r2*np.cos(thetas[0]+thetas[1])+self.r3*np.cos(thetas[0]+thetas[1]+thetas[2]),self.r2*np.cos(thetas[0]+thetas[1])+self.r3*np.cos(thetas[0]+thetas[1]+thetas[2]),self.r3*np.cos(thetas[0]+thetas[1]+thetas[2])]]
        return np.array(J)

    def sudo_inverse_jaco(self,thetas):
        j = self.jacobian(thetas)
        return np.linalg.inv(j.T @ j + self.lambda_damping * np.eye(j.shape[1])) @ j.T
        # return np.linalg.pinv(j)

# The main function which serves as the entry point for the program
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 Python client library
    kinematics_node = Kinematics() # Create an instance of the SimplePublisher

    try:
        rclpy.spin(kinematics_node)  # Keep the node alive and listening for messages
    except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
        pass
    finally:
        kinematics_node.destroy_node()  # Properly destroy the node
        rclpy.shutdown()  # Shutdown the ROS2 Python client library



# This condition checks if the script is executed directly (not imported)
if __name__ == '__main__':
    main()  # Execute the main function
