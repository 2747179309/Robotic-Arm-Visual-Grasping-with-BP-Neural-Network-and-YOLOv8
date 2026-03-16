#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import cv2
import numpy as np
import pyrealsense2 as rs
from pyrealsense2 import depth_frame

from ultralytics import YOLO
import serial
from math import sqrt, atan2, acos, degrees, radians, sin, cos

from pitch_predictor import PitchPredictor


class BusServoController:
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
        frame.append(len(params) + 2)
        frame.append(cmd)
        frame.extend(params)
        self.ser.write(frame)

    def servo_move(self, servos: dict, time_ms: int):
        params = [len(servos), time_ms & 0xFF, (time_ms >> 8) & 0xFF]
        for sid, pos in servos.items():
            pos = int(max(0, min(1000, int(pos))))
            params.extend([int(sid), pos & 0xFF, (pos >> 8) & 0xFF])
        self._send_command(0x03, params)


class FiveDOFKinematics:
    def __init__(self):
        self.link_lengths = {
            'base_height': 5.8,
            'upper_arm': 10.0,
            'forearm': 9.5,
            'wrist_to_gripper': 8.3,
            'gripper_length': 11.4
        }
        self.joint_limits = {
            1: (0, 180),
            2: (-120, 120),
            3: (-120, 120),
            4: (-120, 120),
            5: (-120, 120),
            6: (-120, 120)
        }

    def angle_to_pulse(self, servo_id, angle):
        if servo_id == 1:
            pulse = int(1000 - (1000 / 180.0) * angle)
        else:
            pulse = int(500 + 500.0 * angle / 120.0)
        return max(0, min(1000, pulse))

    def inverse_kinematics(self, x, y, z, pitch, gripper_angle=150, wrist_roll=0):
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

        theta3 = -(pitch - theta5 + theta4)
        theta2 = wrist_roll
        theta1 = gripper_angle

        angles = [theta1, theta2, theta3, theta4, theta5, theta6]
        joint_ids = [1, 2, 3, 4, 5, 6]

        pulses = {}
        for angle, jid in zip(angles, joint_ids):
            pulses[jid] = self.angle_to_pulse(jid, angle)
        return pulses


class ArmPiFPVController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, model_path='pitch_model_py.mat'):
        self.bus = BusServoController(port=port, baudrate=baudrate)
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.pitch_pred = PitchPredictor(model_path)
        self.kin = FiveDOFKinematics()
        self.servo_limits = {
            1: (10, 800),
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

    def move_xyz(self, x, y, z, gripper_angle=150, wrist_roll=45, duration_ms=1500, verbose=True):
        pitch = float(self.pitch_pred.predict(x, y, z))
        servos = self.kin.inverse_kinematics(x, y, z, pitch, gripper_angle=gripper_angle, wrist_roll=wrist_roll)
        servos[3] = servos[3] + 45
        if servos is None:
            raise ValueError("IK failed (unreachable target).")
        servos = self._clip_servos(servos)
        if verbose:
            print(f"[target] x={x:.2f} y={y:.2f} z={z:.2f} cm | pitch={pitch:.3f} deg")
            print("[pulses]", servos)
        self.bus.servo_move(servos, int(duration_ms))
        time.sleep(duration_ms / 1000.0 * 1.1)
        return pitch, servos


class RealSenseYOLOWithDepth:
    def __init__(
        self,
        model_path='bestn.onnx',
        arm_port='/dev/ttyUSB0',
        arm_baudrate=9600,
        pitch_model_path='pitch_model_py.mat',
    ):
        self.model = YOLO(model_path, task='detect')

        self.pipeline = rs.pipeline() #创建数据管线对象，用于取帧
        self.config = rs.config() #配置对象，用来设定要开哪些流（彩色/深度、分辨率、帧率等）。
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline_profile = self.pipeline.start(self.config) #尝试启动管线
        except Exception:
            ctx = rs.context() #创建 RealSense 上下文，用于枚举设备。
            devices = ctx.query_devices() #查询当前连接的设备列表。
            for dev in devices: #for 循环遍历每个设备
                dev.hardware_reset()
            time.sleep(2)
            self.pipeline_profile = self.pipeline.start(self.config)
        #深度对齐
        self.align_to = rs.stream.color #指定对齐到彩色流坐标系。
        self.align = rs.align(self.align_to) #创建对齐器对象，后面用来把 depth 对齐到 color
        #保存深度比例与相机内参
        self.depth_scale = self._get_depth_scale()
        self.intrinsics = self._get_camera_intrinsics()

        self.prev_time = 0 # 初始化上一帧时间戳

        self.arm = ArmPiFPVController(port=arm_port, baudrate=arm_baudrate, model_path=pitch_model_path)
        self._last_cmd_t = 0.0 # 上次触发“询问/动作”的时间（节流用）
        self._moving = False # 一个状态位：是否正在移动（给 _send_arm_target_async 用）
        self._lock = threading.Lock() # 创建互斥锁，用于多线程访问共享变量时防止冲突。

        self.AUTO_CMD_COOLDOWN_S = 1.2
        self.M_TO_CM = 100.0
        self.GRIPPER_ANGLE = 120
        self.GRIPPER_CLOSE_ANGLE = 25
        self.WRIST_ROLL = 45
        self.MOVE_TIME_MS = 1200

        self.FIXED_Z_CM = 20.0
        self.HOME_PULSES = {1: 550, 2: 501, 3: 781, 4: 275, 5: 423, 6: 499}
        self.HOME_OPEN_PULSES = {1: 380, 2: 501, 3: 781, 4: 275, 5: 423, 6: 499}
        self.WAIT2_PULSES = {1: 501, 2: 487, 3: 660, 4: 120, 5: 386, 6: 687}
        self.WAIT1_PULSES = {1: 493, 2: 499, 3: 787, 4: 376, 5: 376, 6: 604}
        self.GRIPPER_STEP_TIME_MS = 800

        self._target_cm = None
        self._prompt_active = False
        self._grab_active = False

    def close(self):
        try:
            self.arm.close()
        except Exception:
            pass

    def _get_depth_scale(self): # _ 表示“内部用”命名约定
        try:
            depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
            return depth_sensor.get_depth_scale()
        except Exception:
            return 0.001

    def _get_camera_intrinsics(self):
        try:
            color_profile = self.pipeline_profile.get_stream(rs.stream.color)
            return color_profile.as_video_stream_profile().intrinsics
        except Exception as e:
            print(f"Fail: {e}")
            return rs.intrinsics(width=640, height=480, fx=615, fy=615, cx=320, cy=240, model=rs.distortion.none)

    def get_average_depth(self, depth_frame, x1, y1, x2, y2):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(depth_frame.width - 1, int(x2))
        y2 = min(depth_frame.height - 1, int(y2))

        depth_data = np.asanyarray(depth_frame.get_data())
        depth_roi = depth_data[y1:y2, x1:x2]

        valid_depth = depth_roi[depth_roi > 0]
        if len(valid_depth) == 0:
            return None

        mean = np.mean(valid_depth)
        std = np.std(valid_depth)
        valid_depth = valid_depth[(valid_depth >= mean - 3 * std) & (valid_depth <= mean + 3 * std)]
        if len(valid_depth) == 0:
            return None

        average_depth = np.mean(valid_depth) * self.depth_scale
        return round(average_depth, 3)

    def pixel_to_3d_xyz(self, depth_frame, pixel_x, pixel_y):# 像素点转物理坐标
        # 利用相机虚拟像平面的相似三角形推导出xyz
        depth_value = depth_frame.get_distance(pixel_x, pixel_y)
        if depth_value <= 0:
            return None
        #利用虚拟像的相似三角形求解x y z=depth_value
        x, y, z = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth_value)
        return round(x, 3), round(y, 3), round(z, 3)

    def _send_arm_target_async(self, x_cm, y_cm, z_cm):
        #这是“异步移动”的封装：开线程跑 self.arm.move_xyz
        #使用线程池（ThreadPoolExecutor），定义worker作为线程函数
        def worker(): #语法点：函数内部可以再定义函数；worker 能访问外层变量（闭包）。
            try:
                self.arm.move_xyz(
                    x_cm, y_cm, z_cm,
                    gripper_angle=self.GRIPPER_ANGLE,
                    wrist_roll=self.WRIST_ROLL,
                    duration_ms=self.MOVE_TIME_MS,
                    verbose=False
                )
            except Exception as e:
                print(f"[arm] {e}")
            finally:
                with self._lock:
                    self._moving = False

        with self._lock:
            if self._moving:
                return
            self._moving = True
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _prompt_grab_async(self, x_cm, y_cm): #检测到目标后，终端询问用户是否抓取
        def worker():
            try:
                ans = input(
                    f"Target detected at x={x_cm:.1f}cm, y={y_cm:.1f}cm, z={self.FIXED_Z_CM:.1f}cm. Grab? (y/n): "
                ).strip().lower()
                if ans in ("y", "yes"):
                    self._grab_sequence_async(x_cm, y_cm)
                else:
                    with self._lock:
                        self._target_cm = None
            finally:
                with self._lock:
                    self._prompt_active = False

        with self._lock:
            if self._prompt_active or self._grab_active:
                return
            self._prompt_active = True
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _grab_sequence_async(self, x_cm, y_cm): #这是“确认抓取后”的动作序列线程
        def worker():
            try:
                self.arm.move_xyz(
                    x_cm, y_cm, self.FIXED_Z_CM,
                    gripper_angle=self.GRIPPER_ANGLE,
                    wrist_roll=self.WRIST_ROLL,
                    duration_ms=2000,
                    verbose=True
                )

                cls = input("Close gripper? (y/n): ").strip().lower()
                if cls in ("y", "yes"):
                    close_pulse = self.arm.kin.angle_to_pulse(1, self.GRIPPER_CLOSE_ANGLE)
                    close_pulse = int(np.clip(close_pulse, self.arm.servo_limits[1][0], self.arm.servo_limits[1][1]))
                    self.arm.bus.servo_move({1: close_pulse}, int(self.GRIPPER_STEP_TIME_MS))
                    time.sleep(self.GRIPPER_STEP_TIME_MS / 1000.0 * 1.1)

                back = input("Return home? (y/n): ").strip().lower()
                if back in ("y", "yes"):
                    self.arm.bus.servo_move(self.HOME_PULSES, 2000)
                    time.sleep(2.2)
                    self.arm.bus.servo_move(self.HOME_OPEN_PULSES, 1200)
                    time.sleep(1.5)
                    self.arm.bus.servo_move({1: close_pulse}, int(self.GRIPPER_STEP_TIME_MS))
                    time.sleep(1)
                    self.arm.bus.servo_move(self.WAIT1_PULSES, 1200)
                    time.sleep(1.5)
                    self.arm.bus.servo_move(self.WAIT2_PULSES, 1200)
                    time.sleep(1.5)            
            except Exception as e:
                print(f"[arm] {e}")
            finally:
                with self._lock:
                    self._grab_active = False
                    self._target_cm = None
                    self._last_cmd_t = time.time()

        with self._lock:
            if self._grab_active:
                return
            self._grab_active = True
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def run(self):
        try:
            while True: #无限循环：每次循环处理一帧
                frames = self.pipeline.wait_for_frames() #阻塞等新帧（RealSense API）
                aligned_frames = self.align.process(frames) #对齐深度到彩色

                color_frame = aligned_frames.get_color_frame() #取彩色帧对象
                depth_frame = aligned_frames.get_depth_frame() #取深度帧对象

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data()) #把彩色帧数据转 numpy 数组

                results = self.model(color_image, verbose=False) #YOLO 推理
                annotated_frame = results[0].plot() #看画框 生成带框的可视化图（annotated_frame）

                best = None
                boxes = results[0].boxes #取所有检测框列表
                for box in boxes: #遍历每个框
                    x1, y1, x2, y2 = box.xyxy[0]

                    pixel_x = int((x1 + x2) / 2)
                    pixel_y = int((y1 + y2) / 2)

                    _ = self.get_average_depth(depth_frame, x1, y1, x2, y2)
                    real_xyz = self.pixel_to_3d_xyz(depth_frame, pixel_x, pixel_y)

                    if real_xyz is not None:
                        real_x, real_y, real_z = real_xyz

                        label_pixel = f"pixel: ({pixel_x}, {pixel_y})"
                        xx = real_x
                        yy = real_y
                        zz = real_z
                        real_x = zz - 0.315
                        real_y = -xx + 0.04
                        real_z = -yy + 0.18
                        label_3d = f"3D: X={real_x:.2f}m, Y={real_y:.2f}m, Z={real_z:.2f}m"

                        cv2.putText(
                            annotated_frame, label_pixel,
                            (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                        )
                        cv2.putText(
                            annotated_frame, label_3d,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )

                        conf = float(box.conf[0]) if hasattr(box, "conf") and box.conf is not None else 0.0

                        x_cm_meas = real_x * self.M_TO_CM
                        y_cm_meas = real_y * self.M_TO_CM
                        z_cm_meas = real_z * self.M_TO_CM

                        if 10.0 <= x_cm_meas <= 17.0 and 10.0 <= abs(y_cm_meas) <= 17.0:
                            dz = abs(z_cm_meas - self.FIXED_Z_CM) #z 与 20cm 的差
                            #三维欧式距离（最近优先）
                            dist = float(np.sqrt(x_cm_meas**2 + y_cm_meas**2 + z_cm_meas**2))
                            score = (dz, dist) #结合成元胞数组 比较时会先比 dz，再比 dist
                            #如果 best 为空或 score 更小，就更新 best（存成字典）
                            if best is None or score < best["score"]:
                                best = {
                                    "conf": conf,
                                    "x": real_x,
                                    "y": real_y,
                                    "z": real_z,
                                    "score": score,
                                    "x_cm_meas": x_cm_meas,
                                    "y_cm_meas": y_cm_meas,
                                    "z_cm_meas": z_cm_meas,
                                }

                now = time.time()
                if best is not None and (now - self._last_cmd_t) >= self.AUTO_CMD_COOLDOWN_S:
                    x_cm = best["x"] * self.M_TO_CM
                    y_cm = best["y"] * self.M_TO_CM
                    z_cm = self.FIXED_Z_CM

                    with self._lock:
                        can_prompt = (self._target_cm is None) and (not self._prompt_active) and (not self._grab_active)

                    if can_prompt:
                        with self._lock:
                            self._target_cm = (x_cm, y_cm, z_cm)
                        self._prompt_grab_async(x_cm, y_cm)
                        self._last_cmd_t = now
                #计算FPS
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
                self.prev_time = curr_time
                cv2.putText(
                    annotated_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )

                cv2.imshow("RealSense YOLOv8 with 3D XYZ", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop() #停相机
            cv2.destroyAllWindows() #关相机窗口
            self.close() #关机械臂串口


if __name__ == "__main__":
    detector = RealSenseYOLOWithDepth(
        model_path='bestn.onnx',
        arm_port='/dev/ttyUSB0',
        arm_baudrate=9600,
        pitch_model_path='pitch_model_py.mat',
    )
    detector.run()
