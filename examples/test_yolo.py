#!/usr/bin/env python3
"""
MAC-SAFE VERSION: YOLO video stream on main thread, robot control on child thread.
Fixes macOS OpenCV crash: "Unknown C++ exception from OpenCV code"

All robot P-control and keyboard teleop is identical to your original file.
"""

import time
import logging
import traceback
import math
import cv2
import numpy as np
import threading
from ultralytics import YOLOE

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# JOINT CALIBRATION
# --------------------------------------------------------------------
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],
    ['shoulder_lift', 2.0, 0.97],
    ['elbow_flex', 0.0, 1.05],
    ['wrist_flex', 0.0, 0.94],
    ['wrist_roll', 0.0, 0.5],
    ['gripper', 0.0, 1.0],
]

def apply_joint_calibration(joint_name, raw_position):
    for j in JOINT_CALIBRATION:
        if j[0] == joint_name:
            return (raw_position - j[1]) * j[2]
    return raw_position

# --------------------------------------------------------------------
# INVERSE KINEMATICS
# --------------------------------------------------------------------
def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    r = math.sqrt(x*x + y*y)
    r_max = l1 + l2
    if r > r_max:
        x *= r_max/r
        y *= r_max/r
        r = r_max
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        x *= r_min/r
        y *= r_min/r
        r = r_min
    cos_t2 = -(r*r - l1*l1 - l2*l2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(cos_t2)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2*math.sin(theta2), l1 + l2*math.cos(theta2))
    theta1 = beta + gamma
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))
    j2 = math.degrees(joint2)
    j3 = math.degrees(joint3)
    j2 = 90 - j2
    j3 = j3 - 90
    return j2, j3

# --------------------------------------------------------------------
# MOVE TO ZERO POSITION
# --------------------------------------------------------------------
def move_to_zero_position(robot, duration=3.0, kp=0.5):
    print("Using P control to slowly move robot to zero position...")

    current_obs = robot.get_observation()
    zero_positions = {
        'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
        'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 0.0
    }

    freq = 50
    steps = int(duration * freq)
    step_time = 1.0 / freq

    for step in range(steps):
        obs = robot.get_observation()
        pos = {k.replace('.pos',''): apply_joint_calibration(k.replace('.pos',''), v)
               for k,v in obs.items() if k.endswith('.pos')}

        act = {}
        for j, t in zero_positions.items():
            cur = pos[j]
            cmd = cur + kp*(t-cur)
            act[j + ".pos"] = cmd

        robot.send_action(act)

        if step % (freq//2)==0:
            print(f"Moving to zero position progress: {100*step/steps:.1f}%")
        time.sleep(step_time)

    print("Robot has moved to zero position")

# --------------------------------------------------------------------
# RETURN TO START POSITION
# --------------------------------------------------------------------
def return_to_start_position(robot, start_positions, kp=0.5, freq=50):
    print("Returning to start position...")

    for _ in range(int(5*freq)):
        obs = robot.get_observation()
        pos = {k.replace('.pos',''): v for k,v in obs.items() if k.endswith('.pos')}
        act = {}
        total_err = 0

        for j, t in start_positions.items():
            cur = pos[j]
            err = t - cur
            total_err += abs(err)
            act[j + ".pos"] = cur + kp*err

        robot.send_action(act)
        if total_err < 2:
            print("Returned to start position")
            break
        time.sleep(1/freq)

    print("Return to start position completed")

# --------------------------------------------------------------------
# ROBOT CONTROL LOOP (CHILD THREAD)
# --------------------------------------------------------------------
def robot_control_loop(robot, keyboard, target_positions, start_positions,
                        current_x, current_y, kp=0.5, freq=50):

    pitch = 0
    pitch_step = 1
    period = 1.0 / freq

    print("Robot P-control loop started (running in child thread)")

    while True:
        try:
            kb = keyboard.get_action()
            if kb:
                for key in kb:
                    if key == "x":
                        print("Exit command detected -> returning to start pos...")
                        return_to_start_position(robot, start_positions, 0.2, freq)
                        return

                    # same logic as your original file
                    if key == "q":
                        target_positions["shoulder_pan"] -= 1
                    if key == "a":
                        target_positions["shoulder_pan"] += 1
                    if key == "t":
                        target_positions["wrist_roll"] -= 1
                    if key == "g":
                        target_positions["wrist_roll"] += 1
                    if key == "y":
                        target_positions["gripper"] -= 1
                    if key == "h":
                        target_positions["gripper"] += 1

                    if key == "w":
                        current_x -= 0.004
                    if key == "s":
                        current_x += 0.004
                    if key == "e":
                        current_y -= 0.004
                    if key == "d":
                        current_y += 0.004

                    # IK update
                    if key in ["w","s","e","d"]:
                        j2,j3 = inverse_kinematics(current_x, current_y)
                        target_positions["shoulder_lift"] = j2
                        target_positions["elbow_flex"] = j3

                    if key == "r":
                        pitch += pitch_step
                    if key == "f":
                        pitch -= pitch_step

            # compute wrist_flex
            target_positions["wrist_flex"] = -target_positions["shoulder_lift"] - target_positions["elbow_flex"] + pitch

            # compute P control
            obs = robot.get_observation()
            pos = {k.replace(".pos",""): apply_joint_calibration(k.replace(".pos",""), v)
                   for k,v in obs.items() if k.endswith(".pos")}

            act = {}
            for j,t in target_positions.items():
                act[j + ".pos"] = pos[j] + kp*(t-pos[j])

            robot.send_action(act)
            time.sleep(period)

        except Exception as e:
            print("Robot loop error:", e)
            traceback.print_exc()
            break

# --------------------------------------------------------------------
# MAIN (YOLO + GUI IN MAIN THREAD)
# --------------------------------------------------------------------
def main():
    print("LeRobot Mac-Safe YOLO + Keyboard Control")
    print("="*60)

    from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
    from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

    port = input("Enter SO100 USB port: ").strip()
    robot = SO100Follower(SO100FollowerConfig(port=port))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    robot.connect()
    keyboard.connect()
    print("Devices connected!")

    # -----------------------------------------
    # CALIBRATION
    # -----------------------------------------
    choice = input("Recalibrate robot? (y/n): ").lower()
    if choice in ["y","yes"]:
        robot.calibrate()
        print("Calibration completed!")
    else:
        print("Using previous calibration file.")

    # -----------------------------------------
    # READ START POSITION
    # -----------------------------------------
    obs = robot.get_observation()
    start_positions = {k.replace(".pos",""): int(v)
                       for k,v in obs.items() if k.endswith(".pos")}
    print("Start positions:", start_positions)

    move_to_zero_position(robot, duration=3.0)

    target_positions = {
        "shoulder_pan":0, "shoulder_lift":0, "elbow_flex":0,
        "wrist_flex":0, "wrist_roll":0, "gripper":0
    }

    current_x, current_y = 0.1629, 0.1131

    # -----------------------------------------
    # YOLO SETUP
    # -----------------------------------------
    print("\nYOLO SETUP\n" + "="*60)
    model = YOLOE("yoloe-11l-seg.pt")

    objs = input("Enter objects to detect: ").strip()
    if objs:
        target_objects = [o.strip() for o in objs.split(",")]
    else:
        target_objects = ["bottle"]
    print("Detecting:", target_objects)

    model.set_classes(target_objects, model.get_text_pe(target_objects))

    # -----------------------------------------
    # CAMERA
    # -----------------------------------------
    cams = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cams.append(i)
        cap.release()

    print("Available cameras:", cams)
    cam_id = int(input("Select camera: "))
    cap = cv2.VideoCapture(cam_id)

    # -----------------------------------------
    # START ROBOT CONTROL THREAD
    # -----------------------------------------
    t = threading.Thread(
        target=robot_control_loop,
        args=(robot, keyboard, target_positions, start_positions,
              current_x, current_y),
        daemon=True)
    t.start()

    # -----------------------------------------
    # MAIN LOOP: YOLO + imshow  (Main thread!)
    # -----------------------------------------
    print("Starting YOLO stream on main thread...")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         continue

    #     results = model(frame)
    #     annotated = results[0].plot()

    #     cv2.imshow("YOLO (Main Thread)", annotated)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord("q"):
    #         break

    print("Starting YOLO stream with IK-based 2D tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)

        if results and results[0].boxes:
            annotated = results[0].plot()

            # ----------------------------------------------
            # --- SELECT FIRST DETECTED OBJECT TO FOLLOW ---
            # ----------------------------------------------
            box = results[0].boxes[0]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            H, W = frame.shape[:2]
            center_x = W / 2
            center_y = H / 2

            # normalized errors in [-1, 1]
            error_x = (cx - center_x) / center_x
            error_y = (cy - center_y) / center_y

            # ----------------------------------------------
            # --- CONVERT VISUAL ERRORS TO IK TARGET SHIFT ---
            # ----------------------------------------------
            gain_x = 0.003   # tune horizontally
            gain_y = 0.003   # tune vertically

            # "target moves right" means "robot moves right"
            current_x -= gain_x * error_x
            current_y += gain_y * error_y  # invert sign for your coordinate frame

            # clamp workspace
            current_x = max(0.05, min(0.25, current_x))
            current_y = max(0.02, min(0.18, current_y))

            # ----------------------------------------------
            # --- UPDATE IK SOLUTION FOR JOINT2+JOINT3 ---
            # ----------------------------------------------
            j2, j3 = inverse_kinematics(current_x, current_y)
            target_positions["shoulder_lift"] = j2
            target_positions["elbow_flex"] = j3

            # overlay debug text
            cv2.putText(annotated, f"ErrX={error_x:+.3f} ErrY={error_y:+.3f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(annotated, f"X={current_x:.3f} Y={current_y:.3f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            annotated = frame

        # show window
        cv2.imshow("YOLO + IK Tracking", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Exit
    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()
    keyboard.disconnect()
    print("Program ended.")


if __name__ == "__main__":
    main()
