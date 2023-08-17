# coding: utf-8
import os
import numpy as np
import pickle
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import time
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cap_dir", required=True, help="path to save capture data")
args = vars(ap.parse_args())

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optinal parameters for registration
# set True if you need
need_bigdepth = True
need_color_depth_map = False

bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
    if need_color_depth_map else None

save_dir = args["cap_dir"]
os.makedirs(save_dir, exist_ok=True)
save_count = 0
while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered,
                       bigdepth=bigdepth,
                       color_depth_map=color_depth_map)

    # NOTE for visualization:
    # cv2.imshow without OpenGL backend seems to be quite slow to draw all
    # things below. Try commenting out some imshow if you don't have a fast
    # visualization backend.
    # cv2.imshow("ir", ir.asarray() / 65535.)
    # cv2.imshow("depth", depth.asarray() / 4500.)
    cv2.imshow("color", cv2.resize(color.asarray(),
                                   (int(1920 / 3), int(1080 / 3))))
    # cv2.imshow("registered", registered.asarray(np.uint8))

    if need_bigdepth:
        cv2.imshow("bigdepth", cv2.resize(bigdepth.asarray(np.float32),
                                          (int(1920 / 3), int(1082 / 3))))
    if need_color_depth_map:
        cv2.imshow("color_depth_map", color_depth_map)#.reshape(424, 512))

    

    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    if key == ord('s'):
        # try:
        cv2.imwrite(os.path.join(save_dir, 'color_%03d.jpg'%save_count), color.asarray(np.uint8))
        with open(os.path.join(save_dir, 'ir_%03d.pkl'%save_count), 'wb') as f:
            pickle.dump(ir.asarray(), f)
        with open(os.path.join(save_dir, 'depth_%03d.pkl'%save_count), 'wb') as f:
            pickle.dump(depth.asarray(), f)
        with open(os.path.join(save_dir, 'big_depth_%03d.pkl'%save_count), 'wb') as f:
            pickle.dump(bigdepth.asarray(np.float32), f)
        with open(os.path.join(save_dir, 'registered_%03d.pkl'%save_count), 'wb') as f:
            pickle.dump(registered.asarray(np.uint8), f)
        print('save data No. %s'%save_count)
        save_count += 1
        # time.sleep(1)
        # except:
        #     print('data save error, try again.')
        #     if save_count > 0:
        #         save_count -= 1
    
    listener.release(frames)

device.stop()
device.close()

sys.exit(0)