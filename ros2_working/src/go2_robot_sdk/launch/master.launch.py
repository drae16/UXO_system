import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node


def generate_launch_description():
    # If the two launch files are in the SAME directory as this master:
    launch_a = os.path.join(get_package_share_directory('go2_robot_sdk'), 'launch', 'robot.launch.py')
    launch_b = os.path.join(get_package_share_directory('interbotix_xsarm_moveit'), 'launch', 'xsarm_moveit.launch.py')


    return LaunchDescription([
        # Include your existing launch files
        IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_a)),

        # Add your two additional nodes here
        Node(
            package='nav_search',
            executable='arm_search',
            name='arm_search',
            output='screen',
            remappings=[('/joint_states', '/vx300s/joint_states')],
        ),

        Node(
            package='nav_search',
            executable='yolo_node.py',
            name='yolo_node',
            output='screen',
        ),

<<<<<<< HEAD
        IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_b),
                                 launch_arguments = {
                                     'robot_model': 'vx300s',
                                     'hardware_type': 'actual',
                                     'use_world_frame': 'false'
                                 }.items()
                                 ),
=======
        Node(
            package='nav_search',
            executable='gps_gate_node',
            name='yolo_node',
            output='screen',
        ),

        
>>>>>>> local_frame_nav_test
    ])