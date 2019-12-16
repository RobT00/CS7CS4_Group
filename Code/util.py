"""
File containing utility functions
"""

import os
import sys
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import models


class Util:
    """
    Utility Class providing helpful functionality
    """

    def __init__(self, files):
        self.files = files

    @staticmethod
    def df_plot(df, columns):
        """
        Use seaborn to create a scatter figure of desired features if 2 or 3 columns selected,
        otherwise create distribution plot (histogram)
        :param df: DataFrame to plot
        :param columns: List of Headers to Plot
        :return:
        """
        # Warning, this is not perfect yet, need to fix data clean before can be tested

        if isinstance(columns, list):
            n_features = len(columns)
        else:
            n_features = 1
            columns = [columns]

        if n_features > 3 or n_features <= 0:
            print("Less Fancy Exception: df_plot can only visualize upto 3 features")
        if all(col not in df for col in columns):
            print("Less Fancy Exception: columns not consistent with DataFrame")
            return

        if n_features == 1:
            # Generate distribution Plot
            feature = columns[0]
            ax = sns.distplot(df[feature].to_numpy())
        else:
            # Generate Scatter Plot
            if n_features == 2:
                feature_x, feature_y = columns
                ax = sns.scatterplot(feature_x, feature_y)
            elif n_features == 3:
                feature_x, feature_y, feature_z = columns
                ax = plt.figure().add_subplot(111, projection="3d")
                ax.scatter(
                    df[feature_x],
                    df[feature_y],
                    df[feature_z],
                    s=50,
                    alpha=0.6,
                    edgecolors="w",
                )
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                ax.set_zlabel(feature_z)

        plt.show()

    @staticmethod
    def find_missing_no(df, columns):
        """
        Print the number of missing values
        :param df: DataFrame
        :param columns: List of columns
        :return:
        """
        missing_val = {}
        print("Number of Missing at each column")
        length_df = len(df)
        for i in columns:
            total_val = df[i].value_counts().sum()
            missing_val[i] = length_df - total_val
        print(missing_val)

    @staticmethod
    def get_data(path):
        """
        Opens specified csv and returns pandas DataFrame
        :param path: Full/Relative CSV file path
        :return: Pandas DataFrame
        """
        if not path:
            raise FileNotFoundError("{} not found".format(path))
        data = pd.read_csv(path)
        data = data.dropna(axis=0, how="all")  # Drop completely empty rows
        return data

    def copy_file(self, src, file_key, dest, cd_dest=True):
        """
        Copy a file to another dictionary
        :param src: Source folder
        :param file_key: Key for file from class instance dictionary
        :param dest: Destination folder
        :param cd_dest: Boolean to cd to destination folder
        :return: Path to copied file
        """
        os.chdir(src)
        a = shutil.copy(self.files.get(file_key), dest)
        if cd_dest:
            os.chdir(dest)
        return a

    def write_predictions(
        self, df, predictions, output_file, data_dir, tmp_dir, label="set_clicked"
    ):
        """
        Function to write predictions to submission files for submitting results to Kaggle
        :param df: DataFrame from submissions file to be written to
        :param predictions: Numpy array of predicted incomes
        :param output_file: Path to submission file
        :param data_dir: Path to directory containing original files
        :param tmp_dir: Path to directory containing modified files - to be modified
        :param label: Name of output column
        :return:
        """
        # Write to test file
        df[label] = predictions
        models.print_df_stats(predictions)
        df.to_csv(output_file, index=False)
        # Write to submission file
        os.chdir(data_dir)
        submission_file = shutil.copy(self.files["submission"], tmp_dir)
        os.chdir(tmp_dir)
        submission_df = self.get_data(submission_file)
        submission_df[label] = predictions
        submission_df.to_csv(submission_file, index=False)

    @staticmethod
    def regression_threshold(y, threshold=0.1):
        """
        Helper function for regression threshold to class handling
        :param y: Regression predictions
        :param threshold: Threshold to classify at
        :return: Classified results
        """
        y[y > threshold] = 1
        y[y < threshold] = 0

        return y.astype(int)

    @staticmethod
    def get_base_model_class():
        """
        Return the constructor for the Base Model Class
        :return: Base Model Class
        """
        return models.ModelClass

    @staticmethod
    def get_multi_model_class():
        """
        Return the constructor for the Multi Model Class
        :return: Multi Model Class
        """
        return models.MultiModelClass

    @staticmethod
    def get_model_from_string(classname):
        """
        Using a provided string argument, try to find the appropriate Model Class
        :param classname: Name of Model Class
        :return: Found Model Class, AttributeError if no Model Class found
        """
        return getattr(sys.modules[__name__].models, classname)

    @staticmethod
    def load_model_from_path(model_path):
        """
        Use a provided path for a saved model to load it
        :param model_path: Path to saved model
        :return: Loaded model
        """
        return pickle.load(open(model_path, "rb"))

    @staticmethod
    def model_save(model, model_name):
        """
        Save a trained model for later use
        :param model: Trained model
        :param model_name: Name of trained model, will be used for file name
        :return:
        """
        try:
            pickle.dump(model, open(f"{model_name}.sav", "wb"))
        except MemoryError as me:
            print(f"MemoryError encountered with pickle: {me}")
            print("Trying joblib")
            try:
                joblib.dump(model, f"{model_name}.sav")
            except MemoryError as me:
                print(f"MemoryError encountered with joblib: {me}")
                print(f"Not saving model: {model_name}.sav")

    """
    Handy list of feature rows
    recommendation_set_id
    user_id
    session_id
    query_identifier
    query_word_count
    query_char_count
    query_detected_language
    query_document_id
    document_language_provided
    year_published
    number_of_authors
    abstract_word_count
    abstract_char_count
    abstract_detected_language
    first_author_id
    num_pubs_by_first_author
    organization_id
    application_type
    item_type
    request_received
    hour_request_received
    response_delivered
    rec_processing_time
    app_version
    app_lang
    user_os
    user_os_version
    user_java_version
    user_timezone
    country_by_ip
    timezone_by_ip
    local_time_of_request
    local_hour_of_request
    number_of_recs_in_set
    recommendation_algorithm_id_used
    algorithm_class
    cbf_parser
    search_title
    search_keywords
    search_abstract
    time_recs_recieved
    time_recs_displayed
    time_recs_viewed
    clicks
    ctr
    set_clicked
    """
