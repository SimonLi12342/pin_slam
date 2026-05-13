#!/usr/bin/env python3
"""
ROS2 Launch file for PIN-SLAM
Automatically plays bag and runs SLAM node together
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Declare arguments
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='/mnt/d/data/ros2_data/Exit_ros2/Exit_ros2.db3',
        description='Path to ROS2 bag file'
    )
    
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='./config/lidar_slam/run.yaml',
        description='Path to PIN-SLAM config file'
    )
    
    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='/velodyne_points',
        description='Point cloud topic name'
    )
    
    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='1.0',
        description='Bag playback rate (1.0 = real-time)'
    )
    
    # Get launch configurations
    bag_path = LaunchConfiguration('bag_path')
    config_path = LaunchConfiguration('config_path')
    topic = LaunchConfiguration('topic')
    rate = LaunchConfiguration('rate')
    
    # ROS2 bag play
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '-r', rate, '--clock'],
        output='screen',
        name='bag_player'
    )
    
    # PIN-SLAM node
    pin_slam_node = ExecuteProcess(
        cmd=['python3', 'pin_slam_ros2.py', config_path, topic],
        output='screen',
        name='pin_slam',
        cwd=os.getcwd()
    )
    
    # RViz2 (optional, can be commented out)
    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', './config/pin_slam_ros2.rviz'],
        output='screen',
        name='rviz2',
        condition=lambda context: os.path.exists('./config/pin_slam_ros2.rviz')
    )
    
    return LaunchDescription([
        bag_path_arg,
        config_path_arg,
        topic_arg,
        rate_arg,
        LogInfo(msg=['Starting PIN-SLAM with bag: ', bag_path]),
        bag_play,
        pin_slam_node,
        rviz_node,  # Uncomment to enable RViz2
    ])
