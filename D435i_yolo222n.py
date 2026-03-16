import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time


class RealSenseYOLOWithDepth:
    def __init__(self, model_path='bestn.onnx'):
        self.model = YOLO(model_path, task='detect')


        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)


        try:
            self.pipeline_profile = self.pipeline.start(self.config)
        except Exception as e:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self.pipeline_profile = self.pipeline.start(self.config)


        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)


        self.depth_scale = self._get_depth_scale()
        self.intrinsics = self._get_camera_intrinsics()

        self.prev_time = 0

    def _get_depth_scale(self):
   
        try:
            depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
            return depth_sensor.get_depth_scale()
        except Exception as e:
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

    def pixel_to_3d_xyz(self, depth_frame, pixel_x, pixel_y):
     
        depth_value = depth_frame.get_distance(pixel_x, pixel_y) 
        if depth_value <= 0:
            return None

     
        x, y, z = rs.rs2_deproject_pixel_to_point(self.intrinsics, [pixel_x, pixel_y], depth_value)

       
        return round(x, 3), round(y, 3), round(z, 3)

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)  

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

              
                results = self.model(color_image, verbose=False)
                annotated_frame = results[0].plot()

               
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                  
                    pixel_x = int((x1 + x2) / 2)
                    pixel_y = int((y1 + y2) / 2)

                  
                    avg_depth = self.get_average_depth(depth_frame, x1, y1, x2, y2)
                 
                    real_xyz = self.pixel_to_3d_xyz(depth_frame, pixel_x, pixel_y)

                    if real_xyz is not None:
                   
                        real_x, real_y, real_z = real_xyz
                     
                        label_pixel = f"pixel: ({pixel_x}, {pixel_y})"
                        label_3d = f"3D: X={real_x}m, Y={real_y}m, Z={real_z}m"

                     
                        cv2.putText(annotated_frame, label_pixel,
                                    (int(x1), int(y1) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(annotated_frame, label_3d,
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

           
                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
                self.prev_time = curr_time
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               
                cv2.imshow("RealSense YOLOv8 with 3D XYZ", annotated_frame)

              
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = RealSenseYOLOWithDepth(model_path='bestn.onnx')
    detector.run()
