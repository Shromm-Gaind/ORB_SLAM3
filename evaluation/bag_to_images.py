#!/usr/bin/env python3
"""Save images from one topic of a ROS 2 bag as PNG files, for clahe_bench.

Usage (source your ROS 2 workspace first):
    python3 bag_to_images.py <bag_folder> <output_folder> [topic] [max_images]

    topic       default /camera/left/image_raw
    max_images  default 300 (the number clahe_bench reads by default)

Example:
    python3 bag_to_images.py ~/bags/oceanicIII ./left_images /camera/left/image_raw 300
    ./clahe_bench ./left_images 2.0 8 gray

The bag is read directly; nothing needs to be playing.
"""

import os
import sys

import cv2
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    bag_path = sys.argv[1]
    out_dir = sys.argv[2]
    topic = sys.argv[3] if len(sys.argv) > 3 else "/camera/left/image_raw"
    max_images = int(sys.argv[4]) if len(sys.argv) > 4 else 300

    reader = rosbag2_py.SequentialReader()
    # Empty storage id: the format (sqlite3 / mcap) is taken from the bag's metadata
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=""),
        rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        print(f"Topic {topic} is not in the bag. Topics found:")
        for name, type_name in sorted(topic_types.items()):
            print(f"  {name}  [{type_name}]")
        return 1
    if topic_types[topic] != "sensor_msgs/msg/Image":
        print(f"Topic {topic} has type {topic_types[topic]}; this script handles sensor_msgs/msg/Image only.")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    bridge = CvBridge()
    saved = 0

    while reader.has_next() and saved < max_images:
        name, data, _stamp = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, Image)
        # Grayscale stays grayscale; any colour layout becomes BGR, which is what imwrite expects
        if msg.encoding in ("mono8", "8UC1"):
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        else:
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.png"), image)
        saved += 1

    if saved == 0:
        print(f"No messages were read from {topic}.")
        return 1

    print(f"Saved {saved} images ({image.shape[1]}x{image.shape[0]}) from {topic} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
