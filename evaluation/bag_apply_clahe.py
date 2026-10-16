#!/usr/bin/env python3
"""
Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to stereo images in a ROS2 bag.

Usage:
    python3 bag_apply_clahe.py <input_bag> <output_bag>
"""

import sys
import os
import numpy as np
import cv2

try:
    from rosbags.rosbag2 import Reader, Writer
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    os.system("pip install rosbags --break-system-packages")
    from rosbags.rosbag2 import Reader, Writer
    from rosbags.typesys import get_typestore, Stores


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_bag> <output_bag>")
        sys.exit(1)

    input_bag = sys.argv[1]
    output_bag = sys.argv[2]

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    image_topics = {'/camera/left/image_raw', '/camera/right/image_raw'}
    msg_count = 0
    img_count = 0

    with Reader(input_bag) as reader:
        with Writer(output_bag, version=9) as writer:
            # Create all connections in the output bag
            conn_map = {}
            for conn in reader.connections:
                new_conn = writer.add_connection(
                    conn.topic,
                    conn.msgtype,
                    typestore=typestore,
                )
                conn_map[conn.id] = new_conn

            for conn, timestamp, rawdata in reader.messages():
                if conn.topic in image_topics:
                    # Deserialize the image
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

                    # Get image dimensions and encoding
                    h = msg.height
                    w = msg.width
                    encoding = msg.encoding

                    # Convert to numpy array
                    if encoding == 'rgb8':
                        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    elif encoding == 'rgba8':
                        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 4)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    elif encoding == 'mono8':
                        gray = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w)
                    elif encoding == 'bgr8':
                        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(h, w, 3)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        # Unknown encoding, pass through
                        writer.write(conn_map[conn.id], timestamp, rawdata)
                        msg_count += 1
                        continue

                    # Apply CLAHE
                    enhanced = clahe.apply(gray)

                    # Update message with mono8 enhanced image
                    msg.encoding = 'mono8'
                    msg.step = w

                    # FIX: Provide a 1D numpy array instead of raw bytes
                    msg.data = enhanced.flatten()

                    # Reserialize
                    new_rawdata = typestore.serialize_cdr(msg, conn.msgtype)
                    writer.write(conn_map[conn.id], timestamp, new_rawdata)
                    img_count += 1
                else:
                    # Pass through non-image messages unchanged
                    writer.write(conn_map[conn.id], timestamp, rawdata)

                msg_count += 1
                if msg_count % 500 == 0:
                    print(f"  Processed {msg_count} messages, {img_count} images enhanced...")

    print(f"\nDone! Processed {msg_count} messages total, {img_count} images enhanced with CLAHE")
    print(f"Output bag: {output_bag}")


if __name__ == "__main__":
    main()
