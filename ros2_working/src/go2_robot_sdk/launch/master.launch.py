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
        IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_b),
                                 launch_arguments = {
                                     'robot_model': 'vx300s',
                                     'hardware_type': 'actual' 
                                 }.items()
                                 ),

        # Add your two additional nodes here
        Node(
            package='nav_search',
            executable='arm_search',
            name='arm_search',
            output='screen',
            # parameters=[...],
            # remappings=[...],
        ),
        Node(
            package='nav_search',
            executable='track_target',
            name='track_target',
            output='screen',
            # parameters=[...],
            # remappings=[...],
        ),

        Node(
            package='nav_search',
            executable='yolo_node.py',
            name='yolo_node',
            output='screen',
            # parameters=[...],
            # remappings=[...],
        ),
    ])