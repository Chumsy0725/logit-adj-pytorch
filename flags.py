from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "cifar10-lt", "Dataset to use.")
flags.DEFINE_string("data_home", "data", "Directory where data files are stored.")
flags.DEFINE_integer("train_batch_size", 128, "Train batch size.")
flags.DEFINE_integer("test_batch_size", 100, "Test batch size.")
flags.DEFINE_boolean("use_lightning", False, "Used to determine whether to use lightning wrappers or not.")


def get_flag(key: str):
    global FLAGS
    FLAGS = flags.FLAGS

    return FLAGS[key].value
