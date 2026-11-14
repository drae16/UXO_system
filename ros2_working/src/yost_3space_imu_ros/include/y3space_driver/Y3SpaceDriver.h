#ifndef _Y3SPACE_DRIVER_H
#define _Y3SPACE_DRIVER_H

#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <y3space_driver/SerialInterface.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>
#include <ctime>
#include <math.h>
#include <y3space_driver/am_shim.h>
#include <std_msgs/msg/int32.hpp>

namespace am
{
//! \brief Yost Labs 3-Space ROS Driver Class
class Y3SpaceDriver: public SerialInterface
{
public:
    //!
    //! Constructor
    //!
    Y3SpaceDriver();
    //!
    //! Destructor
    //!
    ~Y3SpaceDriver();

    void initialize();

    //!
    //! \brief run: runs system
    //!
    void run();
    //!
    //! \brief getSoftwareVersion
    //! \return returns software string version
    //!
    const std::string getSoftwareVersion();
    //!
    //! \brief restoreFactorySettings resets everything
    //!
    void restoreFactorySettings();
    //!
    //! \brief getAxisDirection
    //! \return returns axis directions
    //!
    const std::string getAxisDirection();
    //!
    //! \brief getCalibMode
    //! \return 0 for bias 1 for scale and bias
    //!
    const std::string getCalibMode();
    //!
    //! \brief getMIMode
    //! \return 1 if enabled 0 if disabled
    //!
    const std::string getMIMode();
    //!
    //! \brief startGyroCalibration
    //!
    void startGyroCalibration();
    //!
    //! \brief setMIMode
    //! \param on True to set , False to Unset
    //!
    void setMIMode(bool on);

    /**
    * Poll and receive IMU message from sensor
    * @param imu_msg imu sensor message return
    * @return 0 if success
    */
    int getImuMessage(sensor_msgs::msg::Imu& imu_msg);
    
    void syncCB();

    void initDevice();

    bool onConfigure();

    void heartbeatCB();

    bool onCleanup();

private:
    // ROS Member Variables

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr m_imuPub;  ///< Publisher for IMU messages
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr m_tempPub; ///< Publisher for temperature messages
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr m_rpyPub; ///<Publisher for IMU RPY messages>
    rclcpp::TimerBase::SharedPtr pub_timer_;


    std::string m_port = "/dev/ttyACM0";
    std::string m_mode = "relative";
    std::string m_frame = "yost_imu";
    std::string m_imu_topic = "/y3space/imu";
    std::string m_rpy_topic = "/y3space/rpy";
    bool rpy_degrees_ {false};

    // Sensor-frame -> FLU remap (done in software; this sensor ignores :116).
    // Each entry says which sensor axis feeds FLU x, y, z. Default assumes Yost's
    // factory axes (X right, Y up, Z forward -- a LEFT-handed frame).
    std::vector<std::string> axis_map_param_ {"+z", "-x", "+y"};
    tf2::Matrix3x3 axis_map_;
    double axis_det_ {1.0};
    void buildAxisMap();

    // World-frame heading correction (NWU from the sensor -> ENU per REP-103).
    // yaw_enu = yaw_sensor + world_yaw_offset_deg - magnetic_declination_deg
    double world_yaw_offset_deg_ {90.0};
    double magnetic_declination_deg_ {0.0};   // east-positive, e.g. from NOAA calculator
    tf2::Quaternion q_world_offset_;
    int m_baudrate = 115200;
    int m_timeout = 60000;
    int imu_frequency_ = 400;

    sensor_msgs::msg::Imu imu_msg_;
    
    bool debug_ {false};
    bool magnetometer_enabled_ {false};
    bool tared_ {false};
    double timestamp_offset_;
    rclcpp::Time ros_time_start_;
    double msg_latency_ {0};

    std::pair<rclcpp::Time, rclcpp::Time> reference_time_;
    bool time_synced_ = false;

    std::string getFrequencyMsg(int frequency);
    void getParams();
    geometry_msgs::msg::Vector3 getRPY(geometry_msgs::msg::Quaternion &q);
    geometry_msgs::msg::Quaternion getQuaternion(double roll, double pitch, double yaw);
    geometry_msgs::msg::Quaternion toENU(geometry_msgs::msg::Quaternion q);
    void resetTimeStamp();
    void syncTimeStamp();
		
		template<class T>
		std::vector<T> parseString(const std::string &src, char deliminator = ',')
		{
			std::stringstream ss(src);
			T i;
			std::vector<T> result;
			//Wait for the beginning of the message
			while (ss >> i)
			{
				result.push_back(i);
				if (ss.peek() == ',')
				{
					ss.ignore();
				}
			}
			return result;
		}
			
		template<class T>	
		void printVector(const std::vector<T> &src, const std::string &header = "message")
		{
			std::string message = "";
			std::for_each(src.begin(), src.end(), [&](T x){message += std::to_string(x) + " ";});
			ROS_INFO("%s: %s",header.c_str(),message.c_str());
		}	
				
    double getDegree(double rad);
    void setFrequency();
    void setStreamingSlots();
    void setHeader();
    void setSystemTime();
    void setAxisDirection();
    void setMagnetometer(bool on);
    void tareDevice();
    const std::string getMagnetometerEnabled();
    void setFilterMode();
    
    
    rclcpp::Time getYostRosTime(long sensor_time);
    rclcpp::Time toRosTime(double sensor_time);
    rclcpp::Duration toRosDuration(double sensor_time);
    rclcpp::Time getReadingTime(uint64_t sensor_time);

    void pubTimerCB();
    

    static const std::string logger; ///< Logger tag
    
    static const std::string MODE_ABSOLUTE;
    static const std::string MODE_RELATIVE;
		
    /*
     * Below is a list of commands that can be written via the
     * serialWrite() function to send raw commands to the 3-Space
     * firmware. Unless otherwise noted, these are derived directly
     * from the 3-Space API
     */

    //static const double GRAVITY = 9.8;

    // Orientation Sensor Data Commands
    static constexpr auto GET_TARED_ORIENTATION_AS_QUATERNION_WITH_HEADER = ";0\n";
    static constexpr auto GET_TARED_ORIENTATION_AS_QUATERNION        = ":0\n";
    static constexpr auto GET_TARED_ORIENTATION_AS_EULER_ANGLES      = ":1\n";
    static constexpr auto GET_TARED_ORIENTATION_AS_ROTATION_MATRIX   = ":2\n";
    static constexpr auto GET_TARED_ORIENTATION_AS_AXIS_ANGLE        = ":3\n";
    static constexpr auto GET_TARED_ORIENTATION_AS_TWO_VECTOR        = ":4\n";
    static constexpr auto GET_DIFFERENCE_QUATERNION                  = ":5\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_QUATERNION      = ":6\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_QUATERNION_WITH_HEADER = ";6\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_EULER_ANGLES    = ":7\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_ROTATION_MATRIX = ":8\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_AXIS_ANGLE      = ":9\n";
    static constexpr auto GET_UNTARED_ORIENTATION_AS_TWO_VECTOR      = ":10\n";
    static constexpr auto GET_TARED_TWO_VECTOR_IN_SENSOR_FRAME       = ":11\n";
    static constexpr auto GET_UNTARED_TWO_VECTOR_IN_SENSOR_FRAME     = ":12\n";

    // Corrected Raw Data Commands
    static constexpr auto GET_ALL_CORRECTED_COMPONENT_SENSOR                = ":37\n";
    static constexpr auto GET_CORRECTED_GYRO_RATE                           = ":38\n";
    static constexpr auto GET_CORRECTED_GYRO_RATE_WITH_HEADER               = ";38\n";
    static constexpr auto GET_CORRECTED_ACCELEROMETER_VECTOR                = ":39\n";
    static constexpr auto GET_CORRECTED_ACCELEROMETER_VECTOR_WITH_HEADER    = ";39\n";
    static constexpr auto GET_CORRECTED_COMPASS_VECTOR                      = ":40\n";
    static constexpr auto GET_CORRECTED_LINEAR_ACCELERATION_IN_GLOBAL_SPACE = ":41\n";
    static constexpr auto CORRECT_RAW_GYRO_DATA                             = ":48\n";
    static constexpr auto CORRECT_RAW_ACCEL_DATA                            = ":49\n";
    static constexpr auto CORRECT_RAW_COMPASS_DATA                          = ":50\n";

    // Misc. Raw Data Commands
    static constexpr auto GET_TEMPERATURE_C     = ":43\n";
    static constexpr auto GET_TEMPERATURE_C_W_HEADER     = ";43\n";
    static constexpr auto GET_TEMPERATURE_F     = ":44\n";
    static constexpr auto GET_CONFIDENCE_FACTOR = ":45\n";

    // Uncorrected Raw Data Commands
    static constexpr auto GET_ALL_RAW_COMPONENT_SENSOR_DATA = ":64\n";
    static constexpr auto GET_RAW_GYRO_RATE                 = ":65\n";
    static constexpr auto GET_RAW_ACCEL_DATA                = ";66\n";
    static constexpr auto GET_RAW_COMPASS_DATA              = ":67\n";

    // Streaming Commands
    static constexpr auto SET_REFERENCE_VECTOR_SINGLE				= ":105,1\n";
    static constexpr auto SET_REFERENCE_VECTOR_CONTINUOUS		= ":105,2\n";
    //according to http://yeitechnology.freshdesk.com/support/discussions/topics/1000056170 and the user manual, timestamp command is the following
    static constexpr auto SET_TIME_STAMP_REQUEST				= ";221,2\n";
    // Sets the wired response header to return success byte and timestamp
    static constexpr auto SET_HEADER_TS_SUCCESS				  = ":221,3\n";
    static constexpr auto GET_HEADER_SETTING					  = ":222\n";
    static constexpr auto SET_STREAMING_SLOTS_EULER_TEMP        = ":80,1,43,255,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_EULER_QUATERNION  = ":80,1,0,255,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_QUATERNION_EULER  = ":80,0,1,255,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_EULER             = ":80,1,255,255,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_QUATERNION        = ":80,0,255,255,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_QUATERNION_CORRECTED_GYRO_ACCELERATION_LINEAR_IN_GLOBAL = ":80,0,38,41,255,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_AUTOMODALITY	= ":80,6,7,38,39,40,41,255,255\n";
    // ROS IMU Configurations -----------------------------------------------------------------------------
    /*
    * Most of the above commands are standard command expressions
    * from the API, but below are modified versions for use with
    * this ROS driver
    *
    * TODO: Figure out if you can get the covariance
    */
    static constexpr auto SET_STREAMING_SLOTS_ROS_IMU_RELATIVE  = ";80,0,38,41,43,255,255,255,255\n";
    static constexpr auto SET_STREAMING_SLOTS_ROS_IMU_ABSOLUTE  = ";80,6,38,41,43,255,255,255,255\n";
    // ----------------------------------------------------------------------------------------------------

    static constexpr auto GET_STREAMING_SLOTS                   = ":81\n";
    static constexpr auto SET_STREAMING_TIMING_100_MS           = ":82,100000,0,0\n";
    static constexpr auto SET_STREAMING_TIMING_1000_MS          = ":82,1000000,0,0\n";
    static constexpr auto SET_STREAMING_TIMING_5000_MS          = ":82,5000000,0,0\n";
    static constexpr auto GET_STREAMING_TIMING                  = ":83\n";
    static constexpr auto GET_STREAMING_BATCH                   = ":84\n";
    static constexpr auto START_STREAMING                       = ";85\n";
    static constexpr auto STOP_STREAMING                        = ":86\n";
    static constexpr auto UPDATE_CURRENT_TIMESTAMP              = ":95,0\n";
		static constexpr auto SET_FILTER_MODE_KALMAN								= ":123,1\n";
		static constexpr auto SET_FILTER_MODE_QGRAD								  = ":123,5\n";
	


    // Settings Configuration READ Commands
    static constexpr auto GET_AXIS_DIRECTION           = ":143\n";
    static constexpr auto GET_FILTER_MODE              = ":152\n";
    static constexpr auto GET_EULER_DECOMPOSTION_ORDER = ":156\n";
    static constexpr auto GET_MI_MODE_ENABLED          = ":136\n";
    static constexpr auto GET_MAGNETOMETER_ENABLED     = ":142\n";

    // Settings Configuration WRITE Commands
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_XYZ = ":16,0\n";
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_YZX = ":16,1\n";
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_ZXY = ":16,2\n";
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_ZYX = ":16,3\n";
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_XZY = ":16,4\n";
    static constexpr auto SET_EULER_ANGLE_DECOMP_ORDER_YXZ = ":16,5\n";
    static constexpr auto OFFSET_WITH_CURRENT_ORIENTATION  = ":19\n";
    static constexpr auto TARE_WITH_CURRENT_ORIENTATION    = ":96\n";
    static constexpr auto TARE_WITH_CURRENT_QUATERNION     = ":97\n";
    static constexpr auto SET_MAGNETOMETER_DISABLED        = ":109,0\n";
    static constexpr auto SET_MAGNETOMETER_ENABLED         = ":109,1\n";
    static constexpr auto SET_MI_MODE_ENABLED              = ":112,1\n";
    static constexpr auto SET_MI_MODE_DISABLED             = ":112,0\n";
    static constexpr auto BEGIN_MI_MODE_FIELD_CALIBRATION  = ":114\n";
    static constexpr auto SET_AXIS_DIRECTIONS_ENU          = ":116,8\n";
    static constexpr auto SET_AXIS_DIRECTIONS_FLU          = ":116,19\n";
    static constexpr auto SET_AXIS_DIRECTIONS_RFU          = ":116,1\n";
    static constexpr auto SET_AXIS_DIRECTIONS_DEFAULT      = ":116,5\n";

    // Calibration Commands
    static constexpr auto BEGIN_GYRO_AUTO_CALIB       = ":165\n";
    static constexpr auto SET_CALIB_MODE_BIAS         = ":169,0\n";
    static constexpr auto SET_CALIB_MODE_SCALE_BIAS   = ":169,1\n";
    static constexpr auto GET_CALIB_MODE              = ":170\n";

    // System Commands
    static constexpr auto GET_FIRMWARE_VERSION_STRING = ":223\n";
    static constexpr auto RESTORE_FACTORY_SETTINGS    = ":224\n";
    static constexpr auto SOFTWARE_RESET              = ":226\n";
};
}
#endif //_Y3SPACE_DRIVER_H
