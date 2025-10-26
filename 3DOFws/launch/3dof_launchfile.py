import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os

# Define package name
package_name = '3DOF'
# URDF file path
urdf_relative_path = 'urdf/AssemblyVersion3.2.urdf'
# Rviz config file path
rviz_config_relative_path = 'config/config.rviz'

def generate_launch_description():
    # absolute package path
    package_path = launch_ros.substitutions.FindPackageShare(package=package_name).find(package_name)
    # absolute URDF file path
    urdf_path = os.path.join(package_path, urdf_relative_path)
    # absolute Rviz config file path
    rviz_config_path = os.path.join(package_path, rviz_config_relative_path)
    # print urdf path
    print('URDF Path:', urdf_path)

    # load robot description from URDF file
    with open(urdf_path, 'r') as infp:
        robot_description_content = infp.read()
    
    # define paramater with urdf description
    urdf_parameter = {'robot_description': robot_description_content}
    
    # robot state publisher node
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[urdf_parameter],
        arguments=[urdf_path]
    )
    
    # rviz2 node
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]
    )

    return launch.LaunchDescription([
        robot_state_publisher_node,
        rviz_node
    ])