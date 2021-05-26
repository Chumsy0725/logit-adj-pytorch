"""
Code for converting CIFAR Long Tail Datasets from tfrecord format to npz format.
Example usage:
    $ python -m tfr2npz --src=/path/to/name.tfrecord --dest=/path/to/dest/folder
Note that train and test datasets require separate dest folders for above command and they should exist prior to invoking the command.
"""

from absl import app
from absl import flags
import config as config_dataset
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string("src", None, "The path to the tfrecord file that is needed to be converted.")
flags.DEFINE_string("dest", "data", "The directory in which the converted files are to be stored.")


def _read_and_parse_tf_dataset(file_path: str):
    raw_dataset = tf.data.TFRecordDataset(file_path).prefetch(tf.data.experimental.AUTOTUNE)

    def _proto_parse_function(ex_proto):
        return tf.io.parse_single_example(ex_proto, config_dataset.CIFAR_LT_DATASET_TENSOR_FEATURE_DESCRIPTION)

    return raw_dataset.map(_proto_parse_function)


def _parse_image(encoded_image_record):
    image = tf.io.decode_raw(encoded_image_record, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    image = image * (1. / 255) - 0.5

    return image


def _convert_parsed_dataset_to_np(parsed_dataset):
    data, labels = [], []

    for parsed_record in parsed_dataset:
        encoded_image_record = parsed_record["image/encoded"]
        decoded_image = _parse_image(encoded_image_record)
        image = np.array([decoded_image.numpy()])
        image_class_label = np.array([parsed_record["image/class/label"].numpy()])

        data.append(image)
        labels.append(image_class_label)

    return np.array(data), np.array(labels)


def _get_dataset_name_from_src(src: str):
    return Path(src).stem


def _save_npz_to_dest(dest: str, dataset_name: str, data, labels):
    file_path_data = os.path.join(dest, dataset_name + "_data.npz")
    file_path_labels = os.path.join(dest, dataset_name + "_labels.npz")

    np.savez_compressed(file_path_data, data)
    np.savez_compressed(file_path_labels, labels)


def main(_):
    src: str = FLAGS.src
    dest: str = FLAGS.dest

    parsed_tf_dataset = _read_and_parse_tf_dataset(src)
    np_data, np_labels = _convert_parsed_dataset_to_np(parsed_tf_dataset)
    dataset_name: str = _get_dataset_name_from_src(src)
    _save_npz_to_dest(dest, dataset_name, np_data, np_labels)


if __name__ == "__main__":
    app.run(main)
