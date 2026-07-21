import os
import cv2
import time
import requests
import numpy as np
import sys
import math
from unittest.mock import MagicMock
from datetime import datetime


sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.GLX'] = MagicMock()
sys.modules['OpenGL.EGL'] = MagicMock()

import olympe
from logging import getLogger
import cameratransform as ct
from ultralytics import YOLO
from olympe.messages.camera import camera_capabilities, set_camera_mode, set_photo_mode, take_photo, photo_progress 
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged,GpsLocationChanged,AltitudeChanged,AttitudeChanged
from olympe.features.media import media_created, resource_downloaded, indexing_state, download_media
from olympe.enums.camera import camera_mode, photo_mode, photo_format, photo_file_format
from olympe.messages.ardrone3.Piloting import moveBy, moveTo,TakeOff, Landing
from olympe.messages.gimbal import set_target
from olympe.messages.thermal import set_mode, set_rendering
from olympe.enums.thermal import mode, rendering_mode


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

SKYCTRL_IP = "192.168.53.1"  # SkyController IP instead of drone

DRONE_IP = "192.168.42.1"

SIM_DRONE = "10.202.0.1"

photo_directory = r'/home/drl/Parrot_files/images/raw_images'

detect_directory = r'/home/drl/Parrot_files/images/detections'

model_path = r'/home/drl/poi_dog/ros2_working_ws/src/nav_search/models/pipes.pt'


RGB_PARAMS = {
    "focal_length_px": 3002.1,#3115.7,
    "image_size": (4608, 3456),
    "k1": 0.0245,
    "k2": 0.3682,
    "k3": -0.7625,
    "pos_x": 0.0,
    "pos_y": 0.0,
    "elevation": 0.0,
    "tilt": 0.0,
    "heading": 0.0,
    "roll": 0.0,
}


IR_PARAMS = {
    "focal_length_px": 1.8,            # float, mm — ANAFI Thermal IR values
    "image_size": (160, 120),       # tuple, pixels
    "sensor_size": (2.0, 1.5),      # tuple, mm
    "pos_x": 0.0,
    "pos_y": 0.0,
    "elevation": 0.0,
    "tilt": 0.0,
    "heading": 0.0,
    "roll": 0.0,
    "k1": 0.0,
    "k2": 0.0,
    "k3": 0.0,
}


class Planner:
    def __init__(self,
                 plan_drone,
                 distance_x,
                 distance_y,
                 coverage_per_image_x,
                 coverage_per_image_y):
        
        self.distance_x = distance_x
        self.distance_y =distance_y
        self.coverage_x = coverage_per_image_x
        self.coverage_y = coverage_per_image_y
        self.plan_drone = plan_drone
        self.logger = getLogger("planner logger")

    def check_drone_hovering(self):
        state = self.plan_drone.get_state(FlyingStateChanged)
        state_value = state["state"].value
        return state_value == 2
    
    def wait_drone_hovering(self):
        state = self.plan_drone.get_state(FlyingStateChanged)
        state_value = state["state"].value
        total = 0
        while state_value !=2 :
            self.logger.info(f"Waiting to hover: {state_value}")
            time.sleep(0.1)
            state = self.plan_drone.get_state(FlyingStateChanged)
            state_value = state["state"].value
            continue

        return True

    def move_straight(self,dx):
        if self.wait_drone_hovering():
            try:
                assert self.plan_drone(moveBy(dx,0,0,0)).wait().success()
                self.logger.info(f"moved x by {dx}")
                return True
            except:
                self.logger.error("failed to move in x dir")
                return False
        else:
            self.logger.error("failed to move, drone not hovering")
            return False
    
    def rotate_180(self):
        if self.wait_drone_hovering():
            try:
                assert self.plan_drone(moveBy(0,0,0,np.pi)).wait().success()
                self.logger.info("rotated 180 degrees")
                return True
            except:
                self.logger.error("failed to rotate")
                return False
        else:
            self.logger.error("failed to move, drone not hovering")
            return False

    def move_right (self,dy):

        if self.wait_drone_hovering():
            try:
                assert self.plan_drone(moveBy(0,dy,0,0)).wait().success()
                self.logger.info(f"moved y by {dy}")
                return True
            except:
                self.logger.error("failed to move in y dir")
                return False
        else:
            self.logger.error("failed to move, drone not hovering")
            return False


    def generate_flight_grid(self):
        r = np.ceil(self.distance_x / (0.7 * self.coverage_x))
        c = np.ceil(self. distance_y / (0.7 * self.coverage_y))

        return int(r), int(c)

        


class ImageProcessor:

    def __init__(self,model_path,cam_params: dict):
        self.model = YOLO(model_path)
        self.cam_params = cam_params
        self.trans_cam = None
        self.logger = getLogger("IP logger")


    def setup_camera(self):
        params = self.cam_params
        self.trans_cam = ct.Camera(
            ct.RectilinearProjection(
                focallength_px=params["focal_length_px"],
                image=params["image_size"],
            ),
            ct.SpatialOrientation(
                pos_x_m = params["pos_x"], 
                pos_y_m = params["pos_y"],
                elevation_m=params["elevation"],
                tilt_deg=params["tilt"],
                heading_deg=params["heading"],
                roll_deg=params["roll"],
            ),
            ct.BrownLensDistortion(
                k1=params["k1"],
                k2=params["k2"],
                k3=params["k3"],
            )
        )

    def conduct_inference(self, image):
        centers= []
        results = self.model.predict(image,conf=.6)
        for result in results:
            result.save(filename=f'{detect_directory}/{datetime.now()}.jpg')
            positions = result.boxes.xywh.numpy() 
            for box in positions:
                x = int(box[0])
                y = int(box[1])
                centers.append((x,y))

        return centers
        
    def geolocate_target(self, target_pos, lat, lon, alt,yaw):
        self.trans_cam.elevation_m = alt
        self.trans_cam.heading_deg = yaw
        self.trans_cam.setGPSpos(lat, lon, alt)
        point = self.trans_cam.gpsFromImage(target_pos)
        return point 


    def process_image(self,image,lat,lon,alt,yaw):
        gps_points = []
        centers = self.conduct_inference(image) 
        self.logger.info(f'Nm of targets to geolocate: {len(centers)}')
        if len(centers) > 0:
            for center in centers:

                self.logger.info(f"pos = {center[0], center[1]}")
                
                gps_pos = self.geolocate_target((center[0], center[1]),lat,lon,alt,yaw)
                gps_points.append(gps_pos)
        
            return True ,gps_points
        
        else:
            return False, None
        
    def spacial_locate_target(self, target_pos):
        self.trans_cam.elevation_m = .90
        point = self.trans_cam.spaceFromImage(target_pos)
        return point 


    def indoor_process(self,image):
        gps_points = []
        centers = self.conduct_inference(image) 
        self.logger.info(f'Nm of targets to geolocate: {len(centers)}')
        if len(centers) > 0:
            for center in centers:

                self.logger.info(f"pos = {center[0], center[1]}")
                
                gps_pos = self.spacial_locate_target((center[0], center[1]))
                gps_points.append(gps_pos)
        
            return True ,gps_points
        
        else:
            return False, None
            
        



        



class DroneAnafi:
    
    def __init__(self, flight_altitude,dis_x,dis_y):
        self.drone = olympe.SkyController(SKYCTRL_IP)
        #self.drone = olympe.Drone(SIM_DRONE)
        assert self.drone.connect(timeout=2.0,retry=2)

        self.logger = getLogger("drone logger")
        self.alt = flight_altitude
        self.coverage_x = 2* self.alt * math.tan((75*math.pi/180)/2)
        self.coverage_y = 2* self.alt * math.tan((60*math.pi/180)/2)

        self.flight_planner = Planner(self.drone,distance_x=dis_x ,distance_y=dis_y, coverage_per_image_x=self.coverage_x, coverage_per_image_y = self.coverage_y)

        self.rgb_processor = ImageProcessor(model_path,RGB_PARAMS)
        self.ir_processor = ImageProcessor(model_path, IR_PARAMS)

        self.ir_processor.setup_camera()
        self.rgb_processor.setup_camera()


        if not self.drone.media(indexing_state(state="indexed")).wait(_timeout=60).success():
            raise RuntimeError("Media indexing timed out — check SD card")
        self.logger.info("Media indexed and ready")
        


    def get_altitude_and_yaw(self) -> float:
        state = self.drone.get_state(AltitudeChanged)
        yaw_state = self.drone.get_state(AttitudeChanged)
        yaw = yaw_state["yaw"] 
        yaw_deg = math.degrees(yaw)
        return state["altitude"], yaw_deg # meters above takeoff point


    
    def set_render_thermal(self) -> bool:
        # step 1 — turn on thermal pipeline
        result = self.drone(set_mode(mode=mode.standard)).wait()
        if not result.success():
            self.logger.error("Failed to enable thermal mode")
            return False

        # step 2 — set rendering to thermal only
        result = self.drone(set_rendering(
            mode=rendering_mode.thermal,
            blending_rate=0.0
        )).wait()
        if not result.success():
            self.logger.error("Failed to set thermal rendering")
            return False

        self.logger.info("Thermal rendering ready")
        return True

    def set_render_rgb(self) -> bool:
        # switch back to visible only
        result = self.drone(set_mode(mode=mode.disabled)).wait()
        if not result.success():
            self.logger.error("Failed to disable thermal mode")
            return False
        self.logger.info("RGB rendering ready")
        return True
    
    def set_cam_rgb(self)->bool:
        cam_set = self.drone(set_camera_mode(cam_id=0,value=camera_mode.photo))
        if cam_set.wait().success():
            self.logger.info("RGB camera in photo mode")
        else:
            self.logger.info("RGB camera failed to go to photo mode")
            return False 
        photo_set = self.drone(set_photo_mode(       cam_id = 0,
                                                     mode = photo_mode.single,
                                                     format=photo_format.rectilinear,
                                                     file_format = photo_file_format.jpeg,
                                                     burst="burst_14_over_1s" ,         # not used in single mode
                                                     bracketing="preset_1ev",    # not used in single mode
                                                     capture_interval=0.0, 
                                                     ))
        if photo_set.wait().success():
            self.logger.info("RGB cam params set")
        else:
            self.logger.error("RGB camera failed") 
            return False
        
        if self.set_render_rgb():
            self.logger.info("RGB cam ready for image taking")
        
        else: 
            self.logger.error("RGB camera render set failed") 
            return False
        
        return True
        

    
    def set_cam_ir(self)->bool:
        cam_set = self.drone(set_camera_mode(cam_id=0 ,value=camera_mode.photo))
        if cam_set.wait().success():
            self.logger.info("IR camera in photo mode")
        else:
            self.logger.info("IR  camera failed to go to photo mode")
            return False 
        photo_set = self.drone(set_photo_mode(       cam_id=0,
                                                     mode=photo_mode.single,
                                                     format=photo_format.rectilinear,
                                                     file_format= photo_file_format.jpeg,
                                                     burst="burst_14_over_1s",          # not used in single mode
                                                     bracketing="preset_1ev" ,  # not used in single mode
                                                     capture_interval=0.0, 
                                                     ))
        if photo_set.wait().success():
            self.logger.info("IR cam set up for image taking")
        else:
            self.logger.error("IR camera failed") 
            return False
        
        if self.set_render_thermal():
            self.logger.info("IR cam ready for image taking")
        
        else: 
            self.logger.error("IR camera render set failed") 
            return False
        
        return True
        


    def set_gimbal(self,angle)-> bool:

        gimbal = self.drone(set_target(
                gimbal_id=0,
                control_mode='position',
                yaw_frame_of_reference='none',
                yaw=0.0,
                pitch_frame_of_reference='absolute',
                pitch=angle,
                roll_frame_of_reference='none',
                roll=0.0
            )).wait()
        
        return gimbal.success() 


    def get_gps(self)->tuple:
        gps = self.drone.get_state(GpsLocationChanged)
        lat = gps["latitude"]
        lon = gps["longitude"]
        altitude = gps["altitude"]
        return lat,lon,altitude
    
    def take_photo_rgb(self) -> bool:
        if self.set_cam_rgb():
            # wait for photo to be saved, which gives us the media_id directly
            photo_saved = self.drone(photo_progress(result="photo_saved", _policy="wait"))
            self.drone(take_photo(cam_id=0)).wait()

            if not photo_saved.wait(_timeout=30).success():
                self.logger.error("Photo not saved in time")
                return False, None
            
            state = self.drone.get_state(photo_progress)
            self.logger.info(f"Photo taken")
            status = state[0]["result"].value

            while status != 2:
                state = self.drone.get_state(photo_progress)
                status = state[0]["result"].value
                time.sleep(1)
                continue

            received = photo_saved.received_events()
            if not received:
                self.logger.error("No events received")
                return False, None

            state = self.drone.get_state(photo_progress)
            media_id = state[0]["media_id"]  # 0 = cam_id (RGB)
            img = self._fetch_image(media_id)
            self.logger.info(f"Image collected. Processing...")
            lat,lon,_ = self.get_gps()
            alt,yaw = self.get_altitude_and_yaw()
            feedback, pos =  self.rgb_processor.process_image(img,lat,lon,alt,yaw)

            return feedback,pos
        
        
        else:
            self.logger.error('Failed to take image')
            return False, None 
        


    def take_photo_ir(self):
        if self.set_cam_ir():
            photo_saved = self.drone(photo_progress(result="photo_saved", _policy="wait"))
            self.drone(take_photo(cam_id=1)).wait()
            self.logger.info(f"Photo taken")

            if not photo_saved.wait(_timeout=30).success():
                            self.logger.error("Photo not saved in time")
                            return False, None
            
            state = self.drone.get_state(photo_progress)
            self.logger.info(f"{state}")
            status = state[1]["result"].value

            while status != 2:
                state = self.drone.get_state(photo_progress)
                status = state[1]["result"].value
                time.sleep(1)
                self.logger.error("Waiting to save")
                continue
            

            received = photo_saved.received_events()
            if not received:
                self.logger.error("No events received")
                return False, None

            state = self.drone.get_state(photo_progress)
            media_id = state[1]["media_id"]  # 0 = cam_id (RGB)
            img = self._fetch_image(media_id)

            lat,lon,alt = self.get_gps()

            #feedback, pos =  self.ir_processor.process_image(img,lat,lon,alt)

            #return feedback,pos
        
        else:
            self.logger.error('Failed to take image')
            return False, None
        

    def _fetch_image(self, media_id):

        BASE_URL = "http://192.168.53.1:180"  # SkyController proxies drone REST on port 180
        
        resp = requests.get(
            f"{BASE_URL}/api/v1/media/medias/{media_id}",
            timeout=10
        )
        resp.raise_for_status()
        resources = resp.json().get("resources", [])

        for resource in resources:
            url = f"{BASE_URL}{resource['url']}"
            img_resp = requests.get(url, timeout=10)
            img_resp.raise_for_status()

            jpg_bytes = np.frombuffer(img_resp.content, dtype=np.uint8)
            img = cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)

            if img is None:
                self.logger.error("Failed to decode image")
                continue

            filename = os.path.join(photo_directory, f"{resource['resource_id']}.jpg")
            cv2.imwrite(filename, img)
            self.logger.info(f"Saved: {filename} | shape: {img.shape}")
            
            return img


    def takeoff(self):
        curr_alt ,yaw = self.get_altitude_and_yaw()
        try:
            assert self.drone(TakeOff() >> FlyingStateChanged(state="hovering", _timeout=10)).wait().success()
            self.logger.info('Drone takeoff successful')
        except Exception as e:
            self.logger.info(f'Takeoff failed: {e} ')
            return
        
        dz = -(self.alt - curr_alt)

        self.logger.info(f'current altitude = {curr_alt}')
        #self.logger.info(f'dz = {dz}')

        if self.drone(moveBy(0,0,dz,0) >> FlyingStateChanged(state="hovering", _timeout=10)).wait().success():
            self.logger.info(f"successful takeup to {self.alt} meters")
        
        else:
            self.logger.info('Takeoff achieved, failed to reach desired altitude')
        
        if self.set_gimbal(-90):
            self.logger.info('Gimbal set successfylly')
            time.sleep(3)

        else:
            self.logger.info('Failed to set gimbal')

    def wait_drone_hovering(self)->bool:
        state = self.drone.get_state(FlyingStateChanged)
        state_value = state["state"].value
        while state_value !=2 :
            self.logger.info(f"Waiting to hover: {state_value}")
            time.sleep(0.1)
            state = self.drone.get_state(FlyingStateChanged)
            state_value = state["state"].value
            continue
        return True

    def land(self)->bool:
        if self.wait_drone_hovering():
              assert self.drone(Landing()).wait().success()
              self.logger.info('Landing successful')
              return True
        else:
            self.logger.error('Failure to land, drone must be hovering')
            return False
    
    def execute_scan(self):
        #scan assumes you are starting from right most and bottom most portion of the grid and working left
        dir_counter = -1
        rows, col = self.flight_planner.generate_flight_grid()

        #You will move the drone forward to take a picture at each row in a column, once that column is complete
        # drone will rotate, and shift to the next column. 
        for column in range(col):
            for row in range(rows):
                try:
                    if self.move_straight(self.coverage):
                        self.take_photo_rgb()
                        self.take_photo_ir()
                except: 
                    self.logger.error('Failed scan step')
            
            try:
                    mod = dir_counter ** column 
                    self.flight_planner.execute_movement(self.flight_planner.coverage,"rotate")
                    self.flight_planner.execute_movement(mod*self.flight_planner.coverage,"right")
                    
            except Exception as e: 
                self.logger.error(f'Failed scan step: {e}')



    def take_photo_rgb_test(self) -> bool:
        if self.set_cam_rgb():
            # wait for photo to be saved, which gives us the media_id directly
            photo_saved = self.drone(photo_progress(result="photo_saved", _policy="wait"))
            self.drone(take_photo(cam_id=0)).wait()

            if not photo_saved.wait(_timeout=30).success():
                self.logger.error("Photo not saved in time")
                return False, None
            
            state = self.drone.get_state(photo_progress)
            self.logger.info(f"Photo taken")
            self.logger.info(f"state:{state}")
            status = state[0]["result"].value

            while status != 2:
                state = self.drone.get_state(photo_progress)
                status = state[0]["result"].value
                time.sleep(1)
                continue

            received = photo_saved.received_events()
            if not received:
                self.logger.error("No events received")
                return False, None

            state = self.drone.get_state(photo_progress)
            media_id = state[0]["media_id"]  # 0 = cam_id (RGB)
            img = self._fetch_image(media_id)
            self.logger.info(f"Image collected. Processing...")
            feedback, pos =  self.rgb_processor.indoor_process(img)

            return feedback,pos
        

    def move_straight(self,dx):
        self.flight_planner.move_straight(dx)
        return True

    def move_right(self,dy):
        self.flight_planner.move_right(dy)
        return True
    
    def rotate(self):
        self.flight_planner.rotate_180()
        return True
    
    def disconnect(self):
        self.drone.disconnect()
        return True

    
if __name__ == "__main__":
   pass