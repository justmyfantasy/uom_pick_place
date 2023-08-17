from threading import Thread
import cv2
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

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

class FN2Feed:
    """
    """

    def __init__(self, ):
        self.fn = Freenect2()
        # read camera
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)
        self.serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(self.serial, pipeline=pipeline)
        self.color_cam_param = self.device.getColorCameraParams()
        self.listener = SyncMultiFrameListener(
            FrameType.Color | FrameType.Ir | FrameType.Depth)
        # register listeners
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()
        
        self.registration = Registration(self.device.getIrCameraParams(),
                            self.device.getColorCameraParams())
        self.undistorted = None
        self.registered = None
        self.depth = None
        self.bigdepth = None
        self.color = None

        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)
        bigdepth = Frame(1920, 1082, 4)
        while not self.stopped:
            frames = self.listener.waitForNewFrame()
            color = frames["color"]
            depth = frames["depth"]
            self.registration.apply(color, depth,
                                    undistorted,
                                    registered,
                                    bigdepth=bigdepth)
            # convert to np array
            self.color = color.asarray().copy()
            self.depth = depth.asarray(np.float32).copy()
            self.bigdepth = bigdepth.asarray(np.float32).copy()
            self.registered = registered.asarray(np.uint8).copy()

            self.listener.release(frames)

    def stop(self):
        self.stopped = True
        self.device.stop()
        self.device.close()