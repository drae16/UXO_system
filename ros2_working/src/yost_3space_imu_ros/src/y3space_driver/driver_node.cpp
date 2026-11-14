#include <y3space_driver/Y3SpaceDriver.h>
#include <thread>
#include <memory>
#include <signal.h>
#include <am_utils/am_ros2_utility.h>
#include <cstdlib>

std::string NODE_NAME_ = "Y3SpaceDriver";


std::shared_ptr<am::AMLifeCycle> am::Node::node = nullptr;
std::shared_ptr<Y3SpaceDriver> y3_space_driver = nullptr;




int reset_usb()
{
    ROS_INFO("RESETING THE USB");
    const char *command = "sudo usb_modeswitch -R -v 0x2476 -p 0x1010"; // Replace with actual Vendor and Product IDs
    int returnCode = std::system(command);

    if (returnCode == 0) {
        ROS_INFO("USB port reset successfully.");
    } else {
        ROS_INFO("USB port reset failed.");
    }

    return returnCode;
    ROS_INFO("USB device reset successfully");

    return 0;
}

void signal_handler(int signal)
{

	if(y3_space_driver)
	{
		ROS_INFO("Disconnecting from the serial port");
		y3_space_driver->serialDisConnect();
	}

	//reset_usb();

	rclcpp::shutdown();

	exit(0);
}


int main(int argc, char **argv)
{
	signal(SIGINT, signal_handler);


  	rclcpp::init(argc, argv);
  	
	am::Node::node = std::make_shared<am::AMLifeCycle>(NODE_NAME_);

	y3_space_driver = std::make_shared<Y3SpaceDriver>();

    rclcpp::spin(am::Node::node);

	rclcpp::shutdown();

	return 0;
}
