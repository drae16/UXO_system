import os
from launch import LaunchDescription, launch_description_sources
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, OpaqueFunction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import launch
from colorama import Fore, Back, Style
from launch.substitutions import EnvironmentVariable

import subprocess

        
def generate_launch_description():
    
    yaml_file = get_package_share_directory('y3space_driver')+"/launch/config.yaml";
    
    y3space_driver_node = Node(
                package='y3space_driver',
                executable='y3space_driver',
                name='y3space_imu',
                #output='screen',
                emulate_tty=True,
                #remappings=[
                #  ('/mavros/state', state_topic),
                #  ('/mavros/extended_state', ext_state_topic)
                #],
                parameters=[
                          yaml_file
                    ])
  
    
    
    ld = LaunchDescription()
    ld.add_action(y3space_driver_node)
        
    return ld

if __name__ == '__main__':    
    generate_launch_description()
    
    
    
    
    
