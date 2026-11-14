# ROS Yost Lab 3-Space Driver

## Description
This is a simple driver based on the work done by user [Cagatay](https://github.com/cagataysari). This extension of the original driver designed to enhance the direct usability from the command-line without modification.

Out-of-the-box, it allows you to configure the driver with your port, baudrate, etc. and choose whether to use the driver in relative mode (relative to initial position) or absolute mode (relative to magnetic north and gravitational center).

## Supported Devices:
This driver _should_ work with any of the USB/RS232 versions of the [Yost Labs 3-Space sensors](https://yostlabs.com/3-space-sensors/).

It was tested on the "Watertight" model using its RS232 interface.

## Usage
1. Setup and configure your sensor using terminal commands or the 3-Space sensor suite GUI
2. Use the config file (`config/y3space.yaml`) to choose your port, baudrate, timeout, driver mode ('relative' / 'absolute'), etc.
3. (After building) Launch the node using `roslaunch y3space_driver driver.launch` and the driver will begin to output messages on these topics:
	* **/imu/filtered [sesnsor_msgs/Imu]** : Orientation, Angular Velocity, and Linear Acceleration
	* **/imu/temp [std_msgs/Float64]** : The temperature in deg C as per the sensor

# Debian Package Creation and Sharing

Debian packages are created manually because this repository is not maintained by AutoModality and is not changed often.

```
amros dev build deb --clean --install
```
That will create the debian package and assign a version that will identify the build date/time and host.

The package will be installed on machine for testing and validation.  
Upload to [Cloudsmith|https://cloudsmith.io/~automodality/repos/dev/packages/] for sharing (`dev` first and move to `release` when ready). 
