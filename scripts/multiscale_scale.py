import rospy
from scripts.tools import yaml_parser


def img_callback(self, img_msg):
    try:
        cv_image_in = self.bridge.imgmsg_to_cv2(img_msg, self.encoding)


    try:
        self.image_pub.publish(ros_image_out)
    except Exception as e:
        msg = "Error while trying to publish the enhanced image: {}".format(e)
        rospy.logerr(msg)
        print(msg)
        self.error_flag += 60

    if self.error_flag == 0:
        msg = "DONE!,  FORWARD FPS = {}\n".format(1 / time_forward) + "-" * 50
    else:
        msg = "\nErrors occurred during processing the image\n\t -> error code: {}".format(
            self.error_flag) + "-" * 50

    rospy.loginfo(msg)


if __name__ == "__main__":

    # Ros node initialization
    rospy.init_node("MultiScaleRetinex", anonymous=False)

    # get and set option args
    eng_opt.nThreads = 0  # test code only supports nThreads = 1
    eng_opt.batchSize = 1  # test code only supports batchSize = 1
    eng_opt.serial_batches = True  # no shuffle
    eng_opt.no_flip = True  # no flip

    # Get the ros configs, assuming they are in the config folder which is in the same level of this script
    script_path = Path.cwd()
    config_path = Path(script_path, "configs", "ros_config.yaml")
    ros_opt = yaml_parser(config_path)

    # Create handler
    handler = RosEnGan(engan_opt=eng_opt, opt_ros=ros_opt)

    # Start ros loop
    rospy.spin()

