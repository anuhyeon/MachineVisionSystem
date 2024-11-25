import os
import argparse
import sys
import select

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# manipulator
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState
# from open_manipulator_msgs.srv import SetJointPosition

from open_manipulator_msgs.msg import KinematicsPose, OpenManipulatorState

# RealSense
import cv2
from cv_bridge import CvBridge

## Realsense msg
from sensor_msgs.msg import Image

import pickle

present_joint_angle = [0.0, 0.0, 0.0, 0.0, 0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cv_image = None

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def print_present_values(idx=0):
    print('Joint Angle(Rad): [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]'.format(
        present_joint_angle[0],
        present_joint_angle[1],
        present_joint_angle[2],
        present_joint_angle[3],
        present_joint_angle[4]))
    print('Kinematics Pose(Pose X, Y, Z | Orientation W, X, Y, Z): {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        present_kinematics_pose[0],
        present_kinematics_pose[1],
        present_kinematics_pose[2],
        present_kinematics_pose[3],
        present_kinematics_pose[4],
        present_kinematics_pose[5],
        present_kinematics_pose[6]))

class CapturNode(Node):
    qos = QoSProfile(depth=10)
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    def __init__(self, args):
        super().__init__('capture_node')

        self.args = args
        self.bridge = CvBridge()
        self.idx = 0
        key_value = ''

        # Create joint_states subscriber
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            self.qos)
        # self.joint_state_subscription

        # Create kinematics_pose subscriber
        self.kinematics_pose_subscription = self.create_subscription(
            KinematicsPose,
            'kinematics_pose',
            self.kinematics_pose_callback,
            self.qos)
        # self.kinematics_pose_subscription        

        self.image_subscription = self.create_subscription(
            Image,   # from sensor_msgs.msg import Image 메세지 타입 지정
            'camera/camera/color/image_raw',  # 퍼블리싱 노드에서 발행하는 RGB 이미지 토픽
            self.img_callback,
            self.qos
        )

        self.data = {
            'info': 'poses: kinematics pose/ translation|orientation x y z| w x y z \n\
                images: RGB images \n \
                joints: joint1-4 and gripper',
            'poses': [],
            'joints': [],
            'images': [], 
        }

    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z
    
    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

    def img_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')  # YOLO는 BGR 형식을 처리 가능
        cv2.imshow('RealSense Raw img', self.cv_image)
        cv2.waitKey(1)

    def append_data(self):
        self.data['poses'].append(present_kinematics_pose)
        self.data['joints'].append(present_joint_angle)
        self.data['images'].append(self.cv_image)
        print_present_values()

        print(f'\n\n==========!@!@!@!@!@!@ idx: {self.idx} th !!@!@!@!@!@============\n\n')
        self.idx += 1

    def save_data(self):
        os.makedirs(self.args.save_path, exist_ok=True)
        save_file_path = f'{self.args.save_path}/cal_data.pkl'
        with open(save_file_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f'Complete Calibration File: {save_file_path}')

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="../data/calibration",
    )
    args = parser.parse_args()

    try:
        rclpy.init()
    except Exception as e:
        print(e)

    try:
        node = CapturNode(args)
    except Exception as e:
        print(e)

    key_value = get_key(settings)

                    
    try:
        while(rclpy.ok()):
            rclpy.spin_once(node)

            key_value = get_key(settings)
            if key_value == 'c':
                node.append_data()
            elif key_value == 's':
                node.save_data()
            else:
                if key_value == '\x03':
                    break
                else:
                    print_present_values()
    except Exception as e:
        print(e)
    finally:
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
