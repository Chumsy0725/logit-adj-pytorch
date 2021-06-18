from absl import app
import flags
import dataset.utils as utils_dataset
import trainer
import trainer_lightning_wrapped


def main(_):
    # Retrieve the Dataset
    dataset = utils_dataset.fetch_dataset_from_flags()
    dataset_root = flags.get_flag("data_home")
    use_lightning = flags.get_flag("use_lightning")

    # Call the Trainer
    if not use_lightning:
        trainer.run(dataset, dataset_root)
    else:
        trainer_lightning_wrapped.run(dataset, dataset_root)


if __name__ == "__main__":
    app.run(main)
