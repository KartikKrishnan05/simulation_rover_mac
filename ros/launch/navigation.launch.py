"""Nav2 für den Ackermann-Rover im Büro.

Startet (Gazebo muss bereits laufen: ./run.sh):
  - ros_gz_bridge          Gazebo ↔ ROS 2 (Clock, cmd_vel, Odom, TF, Scans, IMU, Kamera)
  - robot_state_publisher  URDF + Gelenkstellungen → TF
  - EKF                    Rad-Odometrie + Gyroskop → /odom, TF odom → base_footprint
  - scan_merger            vorderer + hinterer Lidar → 360°-Scan für AMCL/SLAM
  - Lokalisierung          map_server + AMCL auf ros/maps/office.yaml
                           (oder slam_toolbox mit slam:=True)
  - Navigation             Planer (Smac Hybrid-A*), Regler (MPPI Ackermann),
                           Behaviors, BT-Navigator, Velocity Smoother, Collision Monitor
  - RViz                   Nav2-Standardansicht + Kamerabild/Marker des VLM-Agenten
  - VLM-Szenarien          person_walker (Person läuft auf Kommando ihren Weg) und
                           episode_recorder (Episoden für ein späteres VLA);
                           der Agent selbst läuft separat: pixi run agent "…"

Die Nodes entsprechen nav2_bringup (Jazzy), werden hier aber direkt gestartet,
damit die Lifecycle-Manager ohne Bond-Überwachung laufen: In der Simulation
lösen kurze Hänger der Sim-Uhr sonst einen Komplettabbruch von Nav2 aus.

Nutzung (in der pixi-Umgebung):
  ros2 launch ros/launch/navigation.launch.py              # fertige Karte + AMCL
  ros2 launch ros/launch/navigation.launch.py slam:=True   # Karte live mit slam_toolbox
  ros2 launch ros/launch/navigation.launch.py rviz:=False
  ros2 launch ros/launch/navigation.launch.py vlm:=False   # ohne Person/Recorder
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml

ROS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAV_NODES = [
    # (Paket, Executable/Name, cmd_vel → cmd_vel_nav umleiten)
    ('nav2_controller', 'controller_server', True),
    ('nav2_smoother', 'smoother_server', False),
    ('nav2_planner', 'planner_server', False),
    ('nav2_behaviors', 'behavior_server', True),
    ('nav2_velocity_smoother', 'velocity_smoother', True),
    ('nav2_collision_monitor', 'collision_monitor', False),
    ('nav2_bt_navigator', 'bt_navigator', False),
    ('nav2_waypoint_follower', 'waypoint_follower', False),
]


def lifecycle_manager(name, nodes):
    return Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name=name,
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': nodes,
            # 0 = keine Bond-Überwachung (siehe Modul-Docstring)
            'bond_timeout': 0.0,
        }],
    )


def generate_launch_description():
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')

    slam = LaunchConfiguration('slam')
    map_yaml = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('rviz')
    use_vlm = LaunchConfiguration('vlm')
    log_level = LaunchConfiguration('log_level')

    params_yaml = RewrittenYaml(
        source_file=params_file,
        param_rewrites={
            # Behavior Trees ohne Spin-Recovery
            'default_nav_to_pose_bt_xml': os.path.join(
                ROS_DIR, 'behavior_trees', 'navigate_to_pose_ackermann.xml'),
            'default_nav_through_poses_bt_xml': os.path.join(
                ROS_DIR, 'behavior_trees', 'navigate_through_poses_ackermann.xml'),
        },
        convert_types=True,
    )
    configured_params = ParameterFile(params_yaml, allow_substs=True)
    sim_time = {'use_sim_time': True}
    tf_remaps = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    ros_args = ['--ros-args', '--log-level', log_level]

    with open(os.path.join(ROS_DIR, 'urdf', 'rover.urdf')) as f:
        robot_description = f.read()

    localization = GroupAction(
        condition=UnlessCondition(slam),
        actions=[
            Node(package='nav2_map_server', executable='map_server', name='map_server',
                 output='screen', remappings=tf_remaps,
                 parameters=[configured_params, sim_time, {'yaml_filename': map_yaml}],
                 arguments=ros_args),
            Node(package='nav2_amcl', executable='amcl', name='amcl',
                 output='screen', parameters=[configured_params, sim_time], remappings=tf_remaps,
                 arguments=ros_args),
            lifecycle_manager('lifecycle_manager_localization', ['map_server', 'amcl']),
        ],
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, 'launch', 'online_sync_launch.py')),
        condition=IfCondition(slam),
        launch_arguments={'use_sim_time': 'True', 'slam_params_file': params_yaml}.items(),
    )

    navigation = [
        Node(package=pkg, executable=name, name=name, output='screen',
             parameters=[configured_params, sim_time], arguments=ros_args,
             remappings=tf_remaps + ([('cmd_vel', 'cmd_vel_nav')] if to_nav else []))
        for pkg, name, to_nav in NAV_NODES
    ]
    navigation.append(
        lifecycle_manager('lifecycle_manager_navigation', [n[1] for n in NAV_NODES]))

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        DeclareLaunchArgument(
            'slam', default_value='False',
            description='True: Karte mit slam_toolbox erstellen statt AMCL auf fertiger Karte'),
        DeclareLaunchArgument(
            'map', default_value=os.path.join(ROS_DIR, 'maps', 'office.yaml'),
            description='Karte (YAML) für map_server/AMCL'),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(ROS_DIR, 'config', 'nav2_params.yaml'),
            description='Nav2-Parameter'),
        DeclareLaunchArgument('rviz', default_value='True', description='RViz starten'),
        DeclareLaunchArgument('vlm', default_value='True',
                              description='person_walker und episode_recorder starten'),
        DeclareLaunchArgument('log_level', default_value='info'),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            parameters=[{
                'config_file': os.path.join(ROS_DIR, 'config', 'bridge.yaml'),
                'use_sim_time': True,
            }],
            output='screen',
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[os.path.join(ROS_DIR, 'config', 'ekf.yaml'), sim_time],
            remappings=[('odometry/filtered', 'odom')],
            output='screen',
        ),

        ExecuteProcess(
            cmd=[sys.executable, os.path.join(ROS_DIR, 'scripts', 'scan_merger.py'),
                 '--ros-args', '-p', 'use_sim_time:=true'],
            name='scan_merger',
            output='screen',
        ),

        *[ExecuteProcess(
            cmd=[sys.executable, os.path.join(ROS_DIR, 'vlm', f'{name}.py'),
                 '--ros-args', '-p', 'use_sim_time:=true'],
            name=name, output='screen', condition=IfCondition(use_vlm))
          for name in ('person_walker', 'episode_recorder')],

        localization,
        slam_toolbox,
        *navigation,

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(ROS_DIR, 'rviz', 'rover_vlm.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(use_rviz),
            output='screen',
        ),
    ])
