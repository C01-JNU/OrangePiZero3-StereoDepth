import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('orangepizero3_stereodepth')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package='orangepizero3_stereodepth',
            executable='stereo_depth_node',
            name='stereo_depth_node',
            output='screen',
            parameters=[params_file]
        )
    ])
