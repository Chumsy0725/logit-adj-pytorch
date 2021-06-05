import tensorflow as tf


CIFAR_LT_DATASET_TENSOR_FEATURE_DESCRIPTION = {
    "image/encoded": tf.io.FixedLenFeature((), tf.string),
    "image/class/label": tf.io.FixedLenFeature([], tf.int64, -1)
}
