#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from pathlib import Path
from tools import CvBr, CvBrComp, yaml_parser, str2bool, RosparamServer
from retinex import MultiScaleRetinex
from dynamic_reconfigure.client import Client
from rretinex.cfg import MSRetinexConfig


class MSRRos:
    def __init__(self, options):

        # Retrieve options
        node_settings = options["Node"]
        retinex_settings = options["Retinex"]
        n_name = node_settings["Name"]

        # image color space
        self.color_space = node_settings["ColorSpace"]

        # Define Subscriber and cv bridge for input image
        self.bridge_in, self.subscriber = self._set_subscriber(node_settings)
        rospy.loginfo("Subscribing to {}".format(self.subscriber.name))

        # Define bridge out and Publisher
        self.bridge_out, self.publisher = self._set_publisher(node_settings)
        rospy.loginfo("Publishing to {}".format(self.publisher.name))

        # Initialize MSR object
        self.MSR = MultiScaleRetinex()
        self.MSR.variances = retinex_settings["Variance"]
        rospy.loginfo("Node {} Ready".format(n_name))

        # get Config
        self.client = Client(n_name, timeout=30, config_callback=self._rosparam_callback)

    def _set_subscriber(self, node_options):
        in_opt = node_options["In"]
        topic_img_sub = in_opt["Topic"]
        compressed = str2bool(in_opt["Compressed"])

        # Define the Cvbridge handler
        if compressed:
            in_br = CvBrComp(ros2cv_encoding=self.color_space)
            subscriber = rospy.Subscriber(name=topic_img_sub, data_class=CompressedImage, callback=self.img_callback)
        else:
            in_br = CvBr(ros2cv_encoding=self.color_space)
            subscriber = rospy.Subscriber(name=topic_img_sub, data_class=Image, callback=self.img_callback)

        return in_br, subscriber

    def _set_publisher(self, node_options):
        out_opt = node_options["Out"]
        topic_img_pub = out_opt["Topic"]
        compressed = str2bool(out_opt["Compressed"])
        encoding_comp_out = out_opt["Format"]

        # Define the Cvbridge handler
        if compressed:
            out_br = CvBrComp(ros2cv_encoding=self.color_space, cv2ros_comp=encoding_comp_out)
            self.bridge_in.cv2ros_comp = encoding_comp_out
            out_topic = "".join([topic_img_pub, "/compressed"])
            publisher = rospy.Publisher(name=out_topic, data_class=CompressedImage, queue_size=1)
        else:
            out_br = CvBr(ros2cv_encoding=self.color_space, cv2ros_encoding=self.color_space)
            self.bridge_in.cv2ros_comp = self.color_space
            publisher = rospy.Publisher(name=topic_img_pub, data_class=Image, queue_size=1)

        return out_br, publisher

    def img_callback(self, img_msg):
        try:
            cv_image_in = self.bridge_in.ros2cv(img_msg)
            enh_image = (self.MSR.do_msr(cv_image_in))
            self.publisher.publish(self.bridge_out.cv2ros(enh_image))

        except Exception as e:
            msg = "Error while trying to publish the enhanced image: {}".format(e)
            rospy.logerr(msg)
            print(msg)

    def _rosparam_callback(self, config):
        variance_string = str(config["variance_list"])
        variance_list = variance_string.split("|")
        if len(variance_list) > 0:
            self.MSR.variance = variance_list
            print("MSR: New variance list: {}".format(variance_list))
        else:
            rospy.logwarn("MSR: Attempt to change the variance list with dynamic reconfigure but the list was empty!")

if __name__ == "__main__":

    # Get the ros configs, assuming they are in the config folder which is in the same level of this script
    script_path = Path(__file__).parent
    config_path = Path(script_path.parent, "config", "config.yaml")
    ros_opt = yaml_parser(config_path)

    # Get the configurations for the single scale retinex
    settings = ros_opt["MultiScaleRetinex"]

    # Node definition
    node_name = settings["Node"]["Name"]
    rospy.loginfo("Initializing Node: {}".format(node_name))
    rospy.init_node(node_name, anonymous=False)

    # Create Server for rosparam
    srv = RosparamServer(config_name=MSRetinexConfig)

    # Create the Retinex Ros Wrapper
    single_retinex = MSRRos(options=settings)

    # ROS!
    rospy.spin()

