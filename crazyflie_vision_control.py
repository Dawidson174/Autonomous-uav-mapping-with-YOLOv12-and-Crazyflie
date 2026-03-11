# -*- coding: utf-8 -*-
#
# ,---------,       ____  _ __
# |  ,-^-,  |      / __ )(_) /_______________ _____  ___
# | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
# Copyright (C) 2023 Bitcraze AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, in version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Example of how to connect to a motion capture system and feed the position to a
Crazyflie, using the motioncapture library. The motioncapture library supports all major mocap systems and provides
a generalized API regardless of system type.
The script uses the high level commander to upload a trajectory to fly a figure 8.

Set the uri to the radio settings of the Crazyflie and modify the
mocap setting matching your system.
"""
import time
from threading import Thread, Lock
import numpy as np
import json
import socket
import csv
import struct
import cv2
import math
import queue
import threading
import argparse

from pynput import keyboard
from termcolor import colored

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.utils.reset_estimator import reset_estimator
from collections import namedtuple

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

host_name = '192.168.102.2'

mocap_system_type = 'qualisys'

rigid_body_name = 'CF2'

send_full_pose = False

orientation_std_dev = 8.0e-3

class MocapWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self._stay_open = True
        self.listen_ip = "192.168.102.2"
        self.listen_port = 12111
        self.rigid_body_name_mapper = 'CF'
        self.pos_mapper = None
        self.pos_AI = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))

        self.start()

    def close(self):
        self._stay_open = False

    def run(self):
        while self._stay_open:
            data, _ = self.sock.recvfrom(2048)
            dt = json.loads(data.decode())
            if rigid_body_name in dt: 
                raw = dt[rigid_body_name]
                self.pos_AI = {
                    'x': raw[1],
                    'y': raw[2],
                    'z': raw[3],
                    'yaw': raw[6]
                    }
                q = euler_to_quaternion(raw[4], raw[5], raw[6])

                if self.on_pose:
                    if send_full_pose:
                        self.on_pose([raw[1], raw[2], raw[3], q])
                    else:
                        self.on_pose([raw[1], raw[2], raw[3], None])

            if self.rigid_body_name_mapper in dt:
                self.pos_mapper = dt[self.rigid_body_name_mapper]


    def get_mapper_pos(self):
        while self._stay_open:
            return self.pos_mapper
        
    def get_AI_pos(self):
        while self._stay_open:
            return self.pos_AI

class UDPStreamer:
    def __init__(self):
        self.listen_ip = "0.0.0.0"
        self.listen_port = 5001
        self.esp_ip = "192.168.4.1"
        self.esp_port = 5000

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))
        self.sock.settimeout(0.01)

        self.cpx_header_size = 4
        self.img_header_magic = 0xBC
        self.img_header_size = 11

        self.buffer = bytearray()
        self.expected_size = None
        self.receiving = False
    
    def start_stream_signal(self):
        try:
            self.sock.sendto(b'FER', (self.esp_ip, self.esp_port))
        except Exception as e:
            print(colored(f"Error starting start signal: {e}", "red"))

    def get_frame(self):
        packet_to_read = 50
        for _ in range(packet_to_read):
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                return None
            except Exception:
                return None
            
            if len(data) >= self.cpx_header_size + 1 and data[self.cpx_header_size] == self.img_header_magic:
                payload = data[self.cpx_header_size:]
                if len(payload) < self.img_header_size:
                    continue

                _, _, _, _, _, size = struct.unpack('<BHHBBI', payload[:self.img_header_size])
                self.expected_size = size
                self.buffer = bytearray(payload[self.img_header_size:])
                self.receiving = True

            elif self.receiving:
                self.buffer.extend(data[self.cpx_header_size:])
                if self.expected_size is not None and len(self.buffer) >= self.expected_size:
                    try:
                        np_data = np.frombuffer(self.buffer, np.uint8)
                        decoded = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                        self.receiving = False
                        self.expected_size = None
                        return decoded
                    except Exception:
                        self.receiving = False
        return None
    def close(self):
        self.sock.close()

class AutoExposure:
    def __init__(self, smoothing = 1.5):
        self.target_brightness = 20
        self.smoothing = smoothing
        self.current_gamma = 1.0

    def process(self, frame):
        if frame is None:
            return None
        
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])

        if brightness < 1:
            brightness = 1
        try:
            new_gamma = math.log(self.target_brightness / 255.0) / math.log(brightness / 255.0)
        except ZeroDivisionError:
            new_gamma = 1.0

        new_gamma = max(0.4, min(new_gamma, 2.5))

        self.current_gamma = (self.current_gamma * (1 - self.smoothing)) + (new_gamma * self.smoothing)

        invGamma = 1.0 / self.current_gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")

        return cv2.LUT(frame, table)

class YoloDetector:
    def __init__(self):
        self.model = YOLO("yolo12s.pt")
        self.names = self.model.names
        self.classes = [2, 15, 32, 58, 41, 65, 67, 11, 74]
        self.conf = 0.40

        self.latest_frame = None
        self.latest_result = []
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.frame_id = 0
        self.processed_id = -1

    def worker(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                fid = self.frame_id

            if frame is None or fid == self.processed_id:
                time.sleep(0.005)
                continue

            results = self.model.predict(frame, conf = self.conf, verbose = False, classes = self.classes)
            detections = []
            if results:
                for r in results:
                    if r.boxes:
                        for box in r.boxes:
                            c = int(box.cls[0])
                            p = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append((x1, y1, x2, y2, c, p))

            with self.lock:
                self.latest_result = detections
                self.processed_id = fid

    def start(self):
        self.running = True
        self.thread = Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def update(self, frame):
        with self.lock:
            self.latest_frame = frame
            self.frame_id += 1

    def get_result(self):
        with self.lock:
            return self.latest_result, self.names

class ObjectManager:
    def __init__(self, dane):
        self.unique_objects = []
        self.dane_zapis = dane

    def update_objects(self, label, gx, gy, gz, drone_pos):
        found_index = -1
        for i, obj in enumerate(self.unique_objects):
            if obj['label'] == label:
                dist = math.sqrt((obj['x'] - gx)**2 + (obj['y'] - gy)**2)
                if dist < 1.0:
                    found_index = i
                    break
        
        if found_index != -1:
            obj = self.unique_objects[found_index]
            obj['x'] = 0.9 * obj['x'] + 0.1 * gx
            obj['y'] = 0.9 * obj['y'] + 0.1 * gy
            obj['z'] = 0.9 * obj['z'] + 0.1 * gz
            obj['count'] += 1
            obj['last_drone_x'] = drone_pos['x']
            obj['last_drone_y'] = drone_pos['y']
            obj['last_drone_z'] = drone_pos['z']
            obj['last_drone_yaw'] = drone_pos['yaw']
        else:
            self.unique_objects.append({
                'label' : label,
                'x' : gx,
                'y' : gy,
                'z' : gz,
                'count' : 1,
                'last_drone_x' : drone_pos['x'],
                'last_drone_y' : drone_pos['y'],
                'last_drone_z' : drone_pos['z'],
                'last_drone_yaw' : drone_pos['yaw']
            })
            print(colored(f"New object: {label} at ({gx:.2f}, {gy:.2f})", "green"))

    def save_to_csv(self):
        try:
            with open(self.dane_zapis, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "drone_x", "drone_y", "drone_z", "drone_yaw", "obj_label", "obj_dist", "obj_angle_deg"])
                f.flush()

                for obj in self.unique_objects:
                    dx = obj['x'] - obj['last_drone_x']
                    dy = obj['y'] - obj['last_drone_y']
                    dist = math.sqrt(dx*dx + dy*dy)

                    global_angle_to_obj = math.atan2(dy, dx)
                    angle_deg_rad = obj['last_drone_yaw'] - global_angle_to_obj
                    angle_deg = math.degrees(angle_deg_rad)

                    writer.writerow([
                        time.time(),
                        obj['last_drone_x'], 
                        obj['last_drone_y'],
                        obj['last_drone_z'],
                        obj['last_drone_yaw'],
                        obj['label'],
                        dist,
                        angle_deg
                    ])

                    f.flush()
        except Exception as e:
            print(colored(f"CSV Save Error: {e}", "red"))

class AiDeckYoloResult:
    def __init__(self, mocap_instance, dane):
        self.streamer = UDPStreamer()
        self.detector = YoloDetector()
        self.auto_exposure = AutoExposure(smoothing=0.05)
        self.obj_manager = ObjectManager(dane)
        self.position = mocap_instance

        self.fov_horizontal = 83.0
        self.ref_heights_at_1m = {41: 37.0, 
                                67: 37.0, 
                                0:  200.0,
                                2: 37.0, 
                                15: 37.0,
                                32: 37.0,
                                58: 37.0,
                                65: 37.0,
                                11: 35.0,
                                74: 50,
                            }
        self.default_height = 200.0

        self.running = False
        self.app_thread = None

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            changed = False
            if key == keyboard.Key.left:
                self.auto_exposure.target_brightness -= 10
                changed = True
            elif key == keyboard.Key.right:
                self.auto_exposure.target_brightness += 10
                changed = True
            
            if changed:
                self.auto_exposure.target_brightness = max(0, min(255, self.auto_exposure.target_brightness))
                print(colored(f"Target Brightness: {self.auto_exposure.target_brightness}", "yellow"))
        except Exception as e:
            print(f"Key error: {e}")

    def loop(self):
        cv2.namedWindow("AI-deck UDP Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI-deck UDP Stream", 1280, 720)

        self.streamer.start_stream_signal()
        self.detector.start()

        last_saved_time = time.time()

        while self.running:
            try:
                frame = self.streamer.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                drone_pos = self.position.get_AI_pos()

                frame_aft = self.auto_exposure.process(frame)

                self.detector.update(frame_aft)

                dets, names = self.detector.get_result()
                img_width = frame_aft.shape[1]
                annotator = Annotator(frame_aft, line_width=2, font_size=10)

                for (x1, y1, x2, y2, cls, conf) in dets:
                    label = names[cls]
                    annotator.box_label([x1, y1, x2, y2], f"{label} {conf:.2f}")

                    cx = (x1 + x2) / 2
                    h_px = y2 - y1
                    if h_px < 5:
                        continue

                    pixel_offset = cx - (img_width / 2)
                    angle_offset_deg = (pixel_offset / img_width) * self.fov_horizontal

                    ref_h = self.ref_heights_at_1m.get(cls, self.default_height)
                    distance = ref_h / h_px

                    global_angle = drone_pos['yaw'] - math.radians(angle_offset_deg)

                    gx = drone_pos['x'] + (distance * math.cos(global_angle))
                    gy = drone_pos['y'] + (distance * math.sin(global_angle))
                    gz = drone_pos['z']

                    self.obj_manager.update_objects(label, gx, gy, gz, drone_pos)
                    cv2.putText(frame_aft, f"{distance:.2f}m", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if time.time() - last_saved_time > 1.0:
                    self.obj_manager.save_to_csv()
                    last_saved_time = time.time()

                final_frame = annotator.result()
                cv2.imshow("AI-deck UDP Stream", final_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(colored("q pressed. Stopping Yolo!!!", "cyan"))
                    self.running = False            
                    break
            except Exception as e:
                print(colored(f"Critical loop error: {e}", "red"))
                time.sleep(0.1)

        self.obj_manager.save_to_csv()
        self.detector.stop()
        self.streamer.close()
        cv2.destroyAllWindows()

    def start(self):
        if self.running:
            return
        self.running = True
        self.app_thread = Thread(target=self.loop, daemon=True)
        self.app_thread.start()

    def stop(self):
        self.running = False
        if self.app_thread:
            self.app_thread.join()

    def is_alive(self):
        return self.running and self.app_thread and self.app_thread.is_alive()

def send_extpose_quat(cf, x, y, z, quat):
    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat.x, quat.y, quat.z, quat.w)
    else:
        cf.extpos.send_extpos(x, y, z)

def adjust_orientation_sensitivity(cf):
    cf.param.set_value('locSrv.extQuatStdDev', orientation_std_dev)

def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    cf.param.set_value('locSrv.extQuatStdDev', 0.06)

def activate_mellinger_controller(cf):
    cf.param.set_value('stabilizer.controller', '2')

def check_battery_voltage(timestamp, data, logconf):
        if 'pm.vbat' in data:
            if float(data['pm.vbat']) >= 3.9:
                print(colored('Battery voltage: ' + str(data['pm.vbat']), 'green'))
            elif 3.1 < float(data['pm.vbat']) < 3.9:
                print(colored('Battery voltage: ' + str(data['pm.vbat']), 'yellow'))
            elif float(data['pm.vbat']) <= 3.1:
                print(colored('Battery voltage: ' + str(data['pm.vbat']), 'red'))


Quaternion = namedtuple('Quaternion', ['w', 'x', 'y', 'z'])

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    return Quaternion(
        w = cr * cp * cy + sr * sp * sy,
        x = sr * cp * cy - cr * sp * sy,
        y = cr * sp * cy + sr * cp * sy,
        z = cr * cp * sy - sr * sp * cy
    )

class AIDroneRun:
    def __init__(self, cf, mocap, data):
        self.running = False
        self.csv_filename = data
        self.cf = cf
        self.mocap = mocap
        self.velocity = 0.2
        self.stop_event = threading.Event()
        self.last_target_pos = None
        self.home_pose = [0, 0, 0.4]
        self.should_stop_loop = False

    def on_press(self, key, cf):
        if key == keyboard.Key.space:
            print(colored("przechwytuje spacje", 'red'))
            self.should_stop_loop = True
            time.sleep(0.2)
            cf.commander.send_stop_setpoint()

    def start_keyboard_listener(self, cf):
        listener = keyboard.Listener(
            on_press=lambda key: self.on_press(key, cf),
        )
        listener.start()
        return listener

    def run_sequence(self, cf):
        commander = cf.high_level_commander

        commander.takeoff(0.4, 2.0)
        time.sleep(3.0)
        try:
            with open(self.csv_filename, 'r') as f:
                pos = f.readlines()
                
                for line in range (1, len(pos)-100, 2):
                    if self.should_stop_loop:
                        break
                    id = line
                    line = pos[line]
                    if line:
                        try:
                            parts = line.strip().split(',')

                            print(f"{id}/{len(pos)}")

                            x = float(parts[2])
                            y = float(parts[3])
                            yaw = float(parts[7])
                                
                            commander.go_to(x, y, 0.4, yaw, 1.0)
                            time.sleep(0.1)
                        except ValueError as e:
                            print(f"Value Error: {e}")
                            continue
                            
                    else:
                        print(colored("[CF2] Lider zakończył zapis. Kończę lot.", "magenta"))
                        

        except FileNotFoundError:
            print(colored("[CF2] BŁĄD: Nie znaleziono pliku CSV!", "red"))

        print(colored("[CF2] Wracam do HOME...", "magenta"))

        commander.land(0.1, 2.0)
        time.sleep(2)
        commander.stop()

    def start(self):
        self.app_thread = Thread(target=self.run_sequence, args=(self.cf,), daemon=True)
        self.app_thread.start()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-fd')
    parser.add_argument('-fd2')
    parser.add_argument('-ff2')

    args = parser.parse_args()

    csv_odczyt = args.fd
    csv_dane = args.fd2
    plik_finished = args.ff2


    print(f'konfiguracja_wczytana: {csv_odczyt}, {csv_dane}, {plik_finished}')

    cflib.crtp.init_drivers()

    mocap_wrapper = MocapWrapper(rigid_body_name)
    Yolo_start = AiDeckYoloResult(mocap_wrapper, csv_dane)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        AIDrone = AIDroneRun(cf, mocap_wrapper, csv_odczyt)

        bat_logger_conf = LogConfig(name='BatCnt', period_in_ms=250)
        bat_logger_conf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(bat_logger_conf)
        bat_logger_conf.data_received_cb.add_callback(check_battery_voltage)

        bat_logger_conf.start()

        listener = AIDrone.start_keyboard_listener(scf.cf)

        mocap_wrapper.on_pose = lambda pose: send_extpose_quat(cf, pose[0], pose[1], pose[2], pose[3])
        activate_kalman_estimator(cf)
        reset_estimator(cf)
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        AIDrone.start()
        Yolo_start.start()

        while AIDrone.app_thread.is_alive():
            time.sleep(0.5)

    mocap_wrapper.close()
    Yolo_start.stop()

    with open(plik_finished, "w") as f:
        f.write(" ")

if __name__ == '__main__':
    main()
