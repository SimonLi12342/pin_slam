"""
ROS2 PointCloud2 creation helper for publishing
"""
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def create_cloud(frame_id, stamp, fields, points):
    """
    Create a PointCloud2 message for ROS2.
    
    Args:
        frame_id: Frame ID string
        stamp: ROS2 Time message
        fields: List of PointField
        points: Nx3 numpy array of points
    
    Returns:
        PointCloud2 message
    """
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp
    
    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = len(points)
    cloud_msg.is_bigendian = False
    cloud_msg.is_dense = True
    cloud_msg.point_step = 12  # 3 floats * 4 bytes
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    cloud_msg.fields = fields
    
    # Pack points as bytes
    buffer = []
    for point in points:
        buffer.append(struct.pack('fff', point[0], point[1], point[2]))
    
    cloud_msg.data = b''.join(buffer)
    
    return cloud_msg
