import yaml
import sys
import argparse
from abc import ABC, abstractmethod
from cv_bridge import CvBridge, CvBridgeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', "True", "Yes", "YES"):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', "False", "No", "NO"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def yaml_parser(yaml_path):
    with open(yaml_path, "r") as stream:
        try:
            config_loaded = yaml.safe_load(stream)
        except yaml.YAMLError:
            msg = "Error while loading the yaml file : {}".format(yaml_path)
            print(msg)
            sys.exit(1)
    return config_loaded


class RosBr(ABC):
    def __init__(self):
        """

        :rtype: object
        """
        self.bridge = CvBridge()

    @abstractmethod
    def ros2cv(self, img_msg):
        pass

    @abstractmethod
    def cv2ros(self, cv_img):
        pass


class CvBr(RosBr):

    def __init__(self, ros2cv_encoding="passthrough", cv2ros_encoding="passthrough"):
        self.ros2cv_encoding = ros2cv_encoding
        self.cv2ros_encoding = cv2ros_encoding

        """
            :param cv2ros_encoding:  The format of the image data, one of the following strings:

               * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
               * from http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#Mat imread(const string& filename, int flags)
               * bmp, dib
               * jpeg, jpg, jpe
               * jp2
               * png
               * pbm, pgm, ppm
               * sr, ras
               * tiff, tif
            """

        super().__init__()

    def ros2cv(self, img_msg):
        try:
            return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding=self.ros2cv_encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            print(msg)

    def cv2ros(self, cv_img):
        try:
            return self.bridge.cv2_to_imgmsg(cv_img, encoding=self.cv2ros_encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            print(msg)


class CvBrComp(RosBr):

    def __init__(self, ros2cv_encoding="passthrough", cv2ros_comp="jpg"):
        self.ros2cv_encoding = ros2cv_encoding
        self.cv2ros_compr = cv2ros_comp
        super().__init__()

    def ros2cv(self, img_msg):
        try:
            return self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding=self.ros2cv_encoding)
        except CvBridgeError as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            print(msg)

    def cv2ros(self, cv_img):

        try:
            return self.bridge.cv2_to_compressed_imgmsg(cv_img, dst_format=self.cv2ros_compr)
        except CvBridgeError as e:
            msg = "Error while trying to convert ROS image to OpenCV: {}".format(e)
            print(msg)
