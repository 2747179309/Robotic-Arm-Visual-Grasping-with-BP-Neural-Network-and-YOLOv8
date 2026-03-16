#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一体化脚本：xyz -> (NN预测pitch) -> 逆解 -> 6舵机脉宽 -> 串口下发控制机械臂

来源合并：
- IK2_ZhengXv (final)222.py ：五自由度/六舵机(含夹爪)逆运动学 + pitch 预测调用
- arm_con_fuben.py ：串口总线舵机通信协议(0x55 0x55 帧头, 0x03 多舵机运动)

依赖：
pip install numpy scipy pyserial
并确保同目录下存在：
- pitch_predictor.py
- pitch_model_py.mat
"""

import os
import sys
import time
import serial
import numpy as np
from math import sqrt, atan2, acos, degrees, radians, sin, cos

from pitch_predictor import PitchPredictor


# ==================== 串口通信协议层 ====================
class BusServoController:
    """总线舵机控制器：发送多舵机位置命令"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def _send_command(self, cmd, params):
        frame = bytearray()
        frame.extend([0x55, 0x55])
        frame.append(len(params) + 2)   # length
        frame.append(cmd)
        frame.extend(params)
        self.ser.write(frame)
        return None

    def servo_move(self, servos: dict, time_ms: int):
        """
        servos: {servo_id(int): pos(0-1000)}
        """
        params = [len(servos), time_ms & 0xFF, (time_ms >> 8) & 0xFF]
        for sid, pos in servos.items():
            pos = int(max(0, min(1000, int(pos))))
            params.extend([int(sid), pos & 0xFF, (pos >> 8) & 0xFF])
        return self._send_command(0x03, params)


# ==================== 运动学逆解（来自 IK2_ZhengXv） ====================
class FiveDOFKinematics:
    """
    ArmPi-FPV 5自由度机械臂（含夹爪共6路舵机）逆运动学
    坐标系：X前，Y左，Z上，单位：cm
    舵机ID：
      ID1 夹爪开合
      ID2 腕部旋转
      ID3 腕部俯仰
      ID4 肘部俯仰
      ID5 肩部俯仰
      ID6 基座旋转
    """
    def __init__(self):
        self.link_lengths = {
            'base_height': 5.8,
            'upper_arm': 10.0,
            'forearm': 9.5,
            'wrist_to_gripper': 8.3,
            'gripper_length': 11.4
        }
        self.joint_limits = {
            1: (0, 180),     # gripper
            2: (-120, 120),  # wrist roll
            3: (-120, 120),  # wrist pitch
            4: (-120, 120),  # elbow
            5: (-120, 120),  # shoulder
            6: (-120, 120)   # base
        }

    def angle_to_pulse(self, servo_id, angle):
        """角度->脉宽(0-1000). 注意：这是你原脚本中的映射。"""
        if servo_id == 1:  # 夹爪
            pulse = int(1000 - (1000 / 180.0) * angle)
        else:
            pulse = int(500 + 500.0 * angle / 120.0)
        return max(0, min(1000, pulse))

    def inverse_kinematics(self, x, y, z, pitch, gripper_angle=150, wrist_roll=0):
        # base rotation
        if y < 0:
            y = -y
            theta6 = degrees(atan2(y, x))
            theta6 = -theta6
        else:
            theta6 = degrees(atan2(y, x))
        r = sqrt(x**2 + y**2)

        end_effector_length = self.link_lengths['wrist_to_gripper'] + self.link_lengths['gripper_length']

        wrist_x = r - end_effector_length * cos(radians(pitch))
        wrist_z = z - self.link_lengths['base_height'] - end_effector_length * sin(radians(pitch))

        d = sqrt(wrist_x**2 + wrist_z**2)

        L2 = self.link_lengths['upper_arm']
        L3 = self.link_lengths['forearm']

        if d > L2 + L3 or d < abs(L2 - L3):
            return None

        try:
            alpha = atan2(wrist_z, wrist_x)
            cos_beta = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
            cos_beta = np.clip(cos_beta, -1, 1)
            beta = acos(cos_beta)
            theta5 = degrees(alpha + beta)
        except Exception:
            return None

        try:
            cos_gamma = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
            cos_gamma = np.clip(cos_gamma, -1, 1)
            gamma = acos(cos_gamma)
            theta4 = degrees(gamma)
        except Exception:
            return None

        # wrist pitch
        theta3 = -(pitch - theta5 + theta4)

        theta2 = wrist_roll
        theta1 = gripper_angle

        angles = [theta1, theta2, theta3, theta4, theta5, theta6]
        joint_ids = [1, 2, 3, 4, 5, 6]

        pulses = {}
        for angle, jid in zip(angles, joint_ids):
            pulse = self.angle_to_pulse(jid, angle)
            pulses[jid] = pulse
        return pulses


# ==================== 合并后的“直接控制”接口 ====================
class ArmPiFPVController:
    """
    一体化控制器：
    - 输入 xyz（cm），自动用神经网络预测 pitch
    - 做 IK 得到 6 舵机脉宽
    - 通过串口下发，让机械臂运动
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, model_path='pitch_model_py.mat'):
        self.bus = BusServoController(port=port, baudrate=baudrate)
        # 模型路径按脚本目录解析，避免工作目录不同导致加载错文件
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.pitch_pred = PitchPredictor(model_path)
        self.kin = FiveDOFKinematics()

        # 安全范围（可按你机械臂实际校准调整）
        self.servo_limits = {
            1: (10, 800),   # 夹爪建议不跑到极限
            2: (0, 1000),
            3: (0, 1000),
            4: (0, 1000),
            5: (0, 1000),
            6: (0, 1000),
        }

    def close(self):
        self.bus.close()

    def _clip_servos(self, servos: dict) -> dict:
        out = {}
        for sid, pos in servos.items():
            mn, mx = self.servo_limits.get(sid, (0, 1000))
            out[sid] = int(np.clip(int(pos), mn, mx))
        return out

    def move_xyz(self, x, y, z, gripper_angle=150, wrist_roll=0, duration_ms=1500, verbose=True):
        """
        输入：x,y,z（cm）
        输出：实际下发的6舵机脉宽（dict），并驱动机械臂运动
        """
        pitch = float(self.pitch_pred.predict(x, y, z))
        servos = self.kin.inverse_kinematics(x, y, z, pitch, gripper_angle=gripper_angle, wrist_roll=wrist_roll)
        if servos is None:
            raise ValueError("逆解失败：目标超出工作空间或姿态不可达。")


        servos = self._clip_servos(servos)

        if verbose:
            print(f"[目标] x={x:.2f} y={y:.2f} z={z:.2f} cm | pitch={pitch:.3f} deg")
            print("[下发] servo pulses:", servos)

        self.bus.servo_move(servos, int(duration_ms))
        time.sleep(duration_ms / 1000.0 * 1.1)
        return pitch, servos


# ==================== 示例主程序 ====================
if __name__ == "__main__":
    # 1) 按需修改串口号：Windows 常见 COM5，Linux 常见 /dev/ttyUSB0
    PORT = "/dev/ttyUSB0"
    BAUD = 9600

    arm = ArmPiFPVController(port=PORT, baudrate=BAUD, model_path="pitch_model_py.mat")
    try:
        print("输入目标 xyz（单位 cm）：")
        x = float(input("x = "))
        y = float(input("y = "))
        z = float(input("z = "))

        # ===================== 分步控制：靠近 -> 到位 -> 夹爪合上 -> 归位 =====================
        # 说明：这里仅补全 main 中的“夹爪合上与归位”流程，不改动前面类/算法实现。
        # 夹爪开合角度与脉宽映射请按实际机械臂标定：
        #   pulse = 1000 - (1000/180)*angle
        # 通常可把 open/close 两个角度做成常量，后续根据实际抓取效果微调。

        GRIPPER_OPEN_ANGLE = 120   # 约对应脉宽 ~500（常用作“张开/半开”），可按实际调整
        GRIPPER_CLOSE_ANGLE = 64   # 约对应脉宽 ~590（常用作“夹紧”），可按实际调整
        APPROACH_DZ = 3.0          # 抓取前上方预靠近高度（cm），避免直接下压碰撞
        HOME_PULSES = {            # 归位脉宽（按你的 arm_con_fuben.py 默认 home_position）
            1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500
        }
        BASKET_PULSES = {
            1: 600, 2: 499, 3: 930, 4: 110, 5: 135, 6: 500
        }

        # 1) 预靠近：先到目标正上方（z + APPROACH_DZ），夹爪张开
        arm.move_xyz(x, y, z + APPROACH_DZ,
                     gripper_angle=GRIPPER_OPEN_ANGLE,
                     wrist_roll=0,
                     duration_ms=2000,
                     verbose=True)

        # 2) 下探到抓取点：夹爪保持张开
        pitch, servos = arm.move_xyz(x, y, z,
                                     gripper_angle=GRIPPER_OPEN_ANGLE,
                                     wrist_roll=45,
                                     duration_ms=1200,
                                     verbose=True)
        cls = input("close?(y/n)")
        if cls == 'y':
            # 3) 夹爪合上（仅控制1号舵机，不改变其余关节）
            close_pulse = arm.kin.angle_to_pulse(1, GRIPPER_CLOSE_ANGLE)
            close_pulse = int(np.clip(close_pulse, arm.servo_limits[1][0], arm.servo_limits[1][1]))
            arm.bus.servo_move({1: close_pulse}, 800)
            time.sleep(0.9)
        else:
            sys.exit()

        back = input("return?(y/n)")
        if back == 'y':
            # 4) 提起：回到目标上方，避免夹紧后拖拽/碰撞
            arm.move_xyz(x, y, z + APPROACH_DZ,
                         gripper_angle=GRIPPER_CLOSE_ANGLE,
                         wrist_roll=0,
                         duration_ms=1200,
                         verbose=True)

            # 5) 归位：回到预设home（脉宽方式），不依赖 IK，保证可执行
            arm.bus.servo_move(HOME_PULSES, 2000)
            time.sleep(2.2)

            # 6) 放置水果到果篮
            arm.bus.servo_move(BASKET_PULSES, 2000)
            time.sleep(2.2)
            arm.bus.servo_move({1: 330}, 800)
            time.sleep(1.5)

            

            print("Finish!")
    finally:
        arm.close()
