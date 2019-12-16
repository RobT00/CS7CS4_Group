"""
Team 66 - Main file
Alec Barber
David Scolard
Robert Trew
"""
import os
import argparse
import shutil
import util


FILE_FORMAT = "csv"

# FILES = {
#     "submission": "tcdml1920-rec-click-pred--submission file.{}".format(FILE_FORMAT),
#     "test": {"full": "tcdml1920-rec-click-pred--test.{}".format(FILE_FORMAT),
#              "blog": "Blog_test.{}".format(FILE_FORMAT),
#              "jabref": "JabRef_test.{}".format(FILE_FORMAT),
#              "myvolts": "MyVolts_test.{}".format(FILE_FORMAT)},
#     "training": {"full": "tcdml1920-rec-click-pred--training.{}".format(FILE_FORMAT),
#                  "blog": "Blog_train.{}".format(FILE_FORMAT),
#                  "jabref": "JabRef_train.{}".format(FILE_FORMAT),
#                  "myvolts": "MyVolts_train.{}".format(FILE_FORMAT)},
# }
FILES = {
    "submission": "tcdml1920-rec-click-pred--submission file.{}".format(FILE_FORMAT),
    "test": "tcdml1920-rec-click-pred--test.{}".format(FILE_FORMAT),
    "training": "tcdml1920-rec-click-pred--training.{}".format(FILE_FORMAT),
}


def run(training, threshold, model_arg, model_save=False, **kwargs):
    """
    Main function used to process data, create models and output predictions
    :param training: Boolean, if the model is being trained or being used to test output
    :param threshold: Threshold to be used for regression to classification predictions
    :param model_arg: Path to a pre-trained model
    :param model_save: Boolean to save trained model(s)
    :return:
    """
    data_man = util.Util(FILES)

    file_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(file_path)
    root_dir = os.path.dirname(script_dir)
    os.chdir(root_dir)
    data_dir = os.path.join(root_dir, "Data")

    tmp_dir = os.path.join(root_dir, "tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    trained_model = False
    try:
        model_class = data_man.get_model_from_string(model_arg)
        model_class = model_class(training=training)  # Instantiate the class instance
        model = (
            model_class.build()
        )  # Construct the model -> can extend to specify params here, if desired
    except AttributeError:
        print(f"Loading model: {model_arg}")
        model = data_man.load_model_from_path(os.path.join(data_dir, model_arg))
        print("Loaded model")
        if type(model) is dict:
            model_class = data_man.get_multi_model_class()
            model_class = model_class(training=training, is_base=True)
        else:
            model_class = data_man.get_base_model_class()
            model_class = model_class(
                training=training
            )  # Instantiate the base model class
        trained_model = True

    # Getting the data upfront
    training_file = data_man.copy_file(data_dir, "training", tmp_dir)
    training_data = data_man.get_data(training_file)
    test_file = data_man.copy_file(data_dir, "test", tmp_dir)
    test_data = data_man.get_data(test_file)

    # Need to get stats for target encoder on test data
    x, y, stats = model_class.process_training(
        training_data,
        is_training=True,
        is_regression=model_class.regression,
        other_df=test_data,
    )
    if not trained_model:
        if training:
            x_train, x_val, y_train, y_val = model_class.ready_training(x, y)
            # Testing SMOTE, a data reducing function for imbalanced datasets
            if not model_class.regression:
                x_train, y_train = model_class.resample_train_data(
                    x_train, y_train, **kwargs
                )
        else:
            if not model_class.regression:
                x, y = model_class.resample_train_data(x, y, **kwargs)

        # Use training data
        if training:
            model = model_class.train(model, x_train, y=y_train)
        else:
            model = model_class.train(model, x, y=y)

        # Save the trained model
        if model_save:
            os.chdir(data_dir)
            data_man.model_save(model, model_class.name)
            os.chdir(tmp_dir)

    if training:
        if trained_model:
            y_pred = model_class.predict(model, x, is_training=training)
            model_class.model_stats(y, y_pred, model_class.regression)
        else:
            y_val_pred = model_class.predict(model, x_val, is_training=training)
            model_class.model_stats(
                y_val, y_val_pred, regression=model_class.regression
            )
    else:
        # Output test predictions
        stats.update(
            {"other_df": training_data}
        )  # Could pass in as another arg, but this is more fun...
        x_test, stats = model_class.process_testing(test_data, stats)
        y_test_pred = model_class.predict(model, x_test, is_training=training)
        if model_class.regression:
            y_test_pred = data_man.regression_threshold(y_test_pred, threshold)

        data_man.write_predictions(
            test_data, y_test_pred.astype(int), test_file, data_dir, tmp_dir
        )

    os.chdir(script_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--training",
        dest="training",
        action="store_true",
        help="Boolean - model is being trained - will generate plots",
        default=False,  # Make True for training
    )
    parser.add_argument(
        "-no-t",
        "--no-training",
        dest="training",
        action="store_false",
        help="Boolean - model is not being trained - no plots",
        default=False,
    )
    parser.add_argument(
        "-th",
        "--threshold",
        dest="threshold",
        help="Threshold to use for regression to classification",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="trained_model",
        help="Model class or path to trained model to use",
        default="CBCMulti",
        type=str,
    )

    args = parser.parse_args()

    run(
        args.training,
        args.threshold,
        args.trained_model,
        **{"over": True, "model_save": False},
    )
