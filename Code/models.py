"""
File for all models, to be used by main.py
"""
import datetime
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, NearMiss
from sklearn.ensemble import RandomForestClassifier
import sklearn.mixture


class ModelClass:
    """
    Placing the process training/testing data here as we can create overrides per model, if needed
    """

    def __init__(self, model_name="Base_Model", training=False):
        base_name = "{}{}".format(
            datetime.datetime.utcnow().strftime("%Y%m%d_%H%M"),
            "_training" if training else "",
        )
        self.name = "_".join([base_name, model_name])
        self.regression = False

    @staticmethod
    def get_total_visits_by_id(df, column="user_id"):
        """
        Utility function for process_training that captures count data from training set.
        This breaks causality, but not in design spec so.....
        :return: dict -- pandas.Series
        """
        df = df[column]
        df = df.dropna()
        df = df[df != "\\N"]
        return df.value_counts()

    def process_training(
        self,
        df,
        is_training=True,
        is_regression=False,
        other_df=None,
        org_id=None,
        precision="float32",
        **stats,
    ):
        """
        Process / manipulate the training data for model training / building
        :param df: DataFrame of training data
        :param is_training: Boolean for whether training or testing is being done
        :param is_regression: Boolean - if regression prediction is being used
        :param other_df: Another DF that can be passed in, i.e. to have access to both training and test data at once
        :param org_id: Id or organisation (explained in body) to allow for per provider processing
        :param precision: Precision at which to store floating point values
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: X (features),
                 y (labels),
                 stats (a dictionary with statistics about the training set that can be applied to the test set)
        """
        id2org = dict({1: "Jabref", 4: "Volts", 8: "Our_blog"})
        print(df.head())
        print(f"org_id: {org_id}")
        print(f"org: {id2org.get(org_id, 'None')}")

        # Create a list of features that we wish to drop. Append as we pass through function
        feature_drop_list = list()
        cols_to_encode = list()

        # Convert DataFrame to Numpy arrays
        _ = df.pop("set_clicked")
        if is_training:
            if is_regression:
                y = df["ctr"].to_numpy(dtype=precision)
            else:
                y = _.to_numpy(dtype=precision)
        else:
            # feature_drop_list.append("set_clicked")
            y = None

        # recommendation_set_id - the unique ID of the set of recommendations. A set comprises of
        # typically around 3-7 items that are displayed/recommended at the same time to a user.
        df["recommendation_set_id"] = df["recommendation_set_id"].fillna(0)
        feature_drop_list.append("recommendation_set_id")

        # user_id - A unique (hashed) ID of a user. Only available for MyVolts. Probably not many
        # users visit the website multiple times and hence this ID is probably of little value for the training.
        df["user_id"] = df["user_id"].fillna("unknown")
        df.loc[df.user_id == "\\N", "user_id"] = "unknown"

        if type(other_df) is pd.DataFrame:  # Make sure we passed in a DataFrame
            test_data_id = self.get_total_visits_by_id(other_df)

            # Create new feature, number of appearances
            train_data_id = df[df["user_id"] != "unknown"]["user_id"].value_counts()
            combined_id = test_data_id.add(train_data_id, fill_value=0)

            df["user_id_num_visits"] = 1
            for id, count in combined_id.items():
                df.loc[df.user_id == id, "user_id_num_visits"] = int(count)

            # Another feature for customer specific processing (only for regular customers)
            df["user_id_person_spefic"] = "unknown"
            df["user_id_person_spefic"] = df[df["user_id_num_visits"] > 5]["user_id"]
            cols_to_encode.append("user_id_person_spefic")

        # Delete now defunct feature
        feature_drop_list.append("user_id")

        # session_id - A unique ID for a session, i.e. one hour or so. Again, only available for
        # MyVolts and it seems to have a bug (some sessions last over days, although it should only be hours).
        df["session_id"] = df["session_id"].fillna("unknown")
        df.loc[df.session_id == "\\N", "session_id"] = "unknown"
        feature_drop_list.append("session_id")

        # query_identifier - The title of the source document. For JabRef, this title is not
        # available due to privacy reasons as the titles represent docum    # Convert Dataframe to Numpy arraysents users have in their personal collections
        df.loc[
            df.query_identifier == "Withheld for privacy", "query_identifier"
        ] = "unknown"
        if org_id != 1:
            feature_drop_list.append("query_identifier")
        else:
            cols_to_encode.append("query_identifier")

        # query_word_count - The number of words in the source document's title
        # NOTE: Perhaps changing to 0 is not best choice - use df_plot(df, ["query_word_count"]) to illustrate.
        df["query_word_count"] = df["query_word_count"].fillna(0)
        df.loc[df.query_word_count == "\\N", "query_word_count"] = 0
        df["query_word_count"] = pd.to_numeric(df["query_word_count"])

        # query_char_count - The number of characters in the source document's title
        df["query_char_count"] = df["query_char_count"].fillna(0)
        df.loc[df.query_char_count == "\\N", "query_char_count"] = 0
        df["query_char_count"] = pd.to_numeric(df["query_char_count"])

        # query_detected_language - We use automatic language detection to detect in which language a document is written
        df["query_detected_language"] = df["query_detected_language"].fillna("unknown")
        df.loc[
            df.query_detected_language == "\\N", "query_detected_language"
        ] = "unknown"
        cols_to_encode.append("query_detected_language")

        # query_document_id - The unique ID of the source document. For MyVolts and the Blog, this should be redundant
        # with the 'query identifier'. However, for some requests for JabRef, there is such an ID given, for others not.
        # If the ID is not given, this means Darwin & Goliath does not have the document in its database and can only use
        # the submitted title for generating recommendations. If the ID is given, it means Darwin & Goliath has the source
        # document in its database and look u information beyond the title (e.g. author names or description/abstract).
        # NOTE: this could be turned to numeric, but dont think that would be useful as is categorical data.

        # TODO: use what unique ids for Jabref that are available in same way that query_identifier uses them.
        if org_id == 1:
            df["query_document_id"] = df["query_document_id"].fillna(1)
            df.loc[df.query_document_id == "\\N", "query_document_id"] = 1
            df.loc[df.query_document_id != 1, "query_document_id"] = -1
        else:
            feature_drop_list.append("query_document_id")

        # document_language_provided -- for some documents, the data owner (JabRef, MyVolts…) provides the document language (not 100% reliable).
        # COMMENT: alot of \N values, probably not worth while, or country tag more useful?
        df.loc[
            df.document_language_provided == "\\N", "document_language_provided"
        ] = "unknown"
        if org_id == 1:
            cols_to_encode.append("document_language_provided")
        else:
            feature_drop_list.append("document_language_provided")

        # year_published -- The year in which the source items was released/published
        # Large number of \N values again....
        if org_id == 1:
            df.loc[df.year_published == "\\N", "year_published"] = pd.to_numeric(
                df[df["year_published"] != "\\N"]["year_published"]
            ).mean()
            df["year_published"] = pd.to_numeric(df["year_published"])
        else:
            feature_drop_list.append("year_published")

        # number_of_authors -- In case of JabRef and research articles, this field describes the number of authors of the source document
        if org_id == 1:
            df.loc[df.number_of_authors == "\\N", "number_of_authors"] = pd.to_numeric(
                df[df["number_of_authors"] != "\\N"]["number_of_authors"]
            ).mean()
            df["number_of_authors"] = pd.to_numeric(df["number_of_authors"])
        else:
            feature_drop_list.append("number_of_authors")

        # abstract_word_count -- The number of words of the description/abstract of the source item
        if org_id == 1 or org_id == 4 or org_id == 8:
            df.loc[
                df.abstract_word_count == "\\N", "abstract_word_count"
            ] = pd.to_numeric(
                df[df["abstract_word_count"] != "\\N"]["abstract_word_count"]
            ).mean()
            df["abstract_word_count"] = pd.to_numeric(df["abstract_word_count"])
        else:
            feature_drop_list.append("abstract_word_count")

        # abstract_char_count -- The number of characters of the description/abstract of the source item
        feature_drop_list.append("abstract_char_count")

        # abstract_detected_language -- The automatically detected language of the source document's abstract
        if org_id == 1:
            cols_to_encode.append("abstract_detected_language")
        else:
            feature_drop_list.append("abstract_detected_language")

        # first_author_id -- The unique ID of the first author of the source document
        # Maybe group all nans together along with all authors that only appear a few times (already encapulated
        # in num_pubs_by_first_author )
        df["first_author_id"] = df["first_author_id"].replace("\\N", 0)
        df["first_author_id"] = df["first_author_id"].replace("\\N", 0)
        counts = df["first_author_id"].value_counts()
        df.loc[
            df["first_author_id"].isin(counts.index[counts < 5]), "first_author_id"
        ] = 0
        # TODO: this has potential, requires using other df again...maybe
        feature_drop_list.append("first_author_id")

        # num_pubs_by_first_author -- The number of documents in Darwin & Goliath database published by the source
        # document's first author. Actually, we have an algorithm that recommends documents by the 'same author'.
        # So, if a source cols_to_encode.append("item_type")document is published by 'author X', the 'same author' algorithm recommends other documents
        # authored by thatcols_to_encode.append("item_type") person. Potentially, the number of documents that the person has authored is a good predictor
        # of whether the 'cols_to_encode.append("item_type")same author' algorithm will perform well.
        # Unsurprisingly mcols_to_encode.append("item_type")ostly \\N, still could be useful though
        # lets use ordinal
        df["num_pubs_by_first_author"] = df["num_pubs_by_first_author"].replace(
            "\\N", 1
        )
        df["num_pubs_by_first_author"] = pd.to_numeric(df["num_pubs_by_first_author"])
        if org_id == 4 or org_id == 8:
            feature_drop_list.append("num_pubs_by_first_author")
        df["num_pubs_by_first_author"] = np.log(df["num_pubs_by_first_author"])

        # organization_id -- the ID of the recommendation partner, i.e. JabRef, MyVolts or our Blog
        # Useful, encode
        if org_id is None:
            cols_to_encode.append("organization_id")
        else:
            feature_drop_list.append("organization_id")

        # application_type -- Digital Library, E-Commerce, or Blog. This should 100% correlate with 'organizationid.
        # VOID Organization_id doesnt have any unknowns, so should drop this
        feature_drop_list.append("application_type")

        # item_type -- For JabRef, this value is always 'academicpublication', for our Blog, this value is always 'article'.
        # For MyVolts, the value differs (e.g. 'Hard drives & NAS' or 'Music making & pedals').
        df["item_type"] = df["item_type"].replace("\\N", np.nan)
        if org_id == 4 or org_id is None:
            cols_to_encode.append("item_type")
        else:
            feature_drop_list.append("item_type")

        # request_received -- The local Irish time when the request for recommendations was received
        # COMMENT: hard to parse, '03/12/2018 16:28' format, not sure if this is usefull when we have hour_request_received which should
        # be more relevant
        feature_drop_list.append("request_received")

        # hour_request_received -- The hour when the request for recommendations was received
        # This is similar to local_hour_of_request, but appears more complete, should test to see if same.
        df["hour_request_received"] = pd.to_numeric(df["hour_request_received"])
        # Create two new feature rows to capture cyclic nature of time
        df["hour_request_received_sin"] = np.sin(
            2 * np.pi * df.hour_request_received / 24
        )
        df["hour_request_received_cos"] = np.cos(
            2 * np.pi * df.hour_request_received / 24
        )
        # Drop original time as full encapsulated by above
        feature_drop_list.append("hour_request_received")

        # response_delivered -- The time when the server returned recommendations. If this time is long (a few seconds) after
        # the request_received, chances are users have closed the web page already, and won't see any recommendations. This
        # feature is not available in the test dataset. This feature can only help you to filter data but not to train a model!!!
        # Cant change... also will require some string processing to be useful (format ~ 21/12/2018 20:34)
        feature_drop_list.append("response_delivered")

        # rec_processing_time -- The duration in seconds it took the server to calculate recommendations. This should be equal
        # to the difference of request_received and response_delivered.
        # Looks fine, but yea, probably absolutely irrevelant
        feature_drop_list.append("rec_processing_time")

        # app_version -- The version of the application that requested recommendations. Probably nA for MyVolts
        # and the Blog but given for JabRef.
        # NOTE: This has a LOT of unique values, not sure if it is useufl enough info to make it worth while using one hot...?
        if org_id == 1:
            cols_to_encode.append("app_version")
        else:
            feature_drop_list.append("app_version")

        # app_lang -- The language of the application that requested recommendations (again, nA for MyVolts
        # and the Blog, but given for JabRef)
        # Is this usefull info? I guess if they are german but speak english they may be less likely to read another article?
        df["app_lang"] = df["app_lang"].replace("\\N", np.nan)
        if org_id == 1 or org_id is None:
            cols_to_encode.append("app_lang")
        else:
            feature_drop_list.append("app_lang")

        # user_os -- The operating system of the user that recommendations are given to
        # Limited use... few useful values
        # Maybe use regex to split: {Windows, Linux, Mac OS, other}  ?
        os_lookups = {
            "^.*(OS X)": "Apple",
            "^Mac OS.*": "Apple",
            "^Wind.*": "Windows",
            "^iOS.*": "iOS",
            "^Andr.*": "Android",
            "^Chro.*": "Chrome OS",
            "^.*.(u).*": "Linux",
            "^Lin.*": "Linux",
            "^Ubu.*": "Linux",
            "^Fed.": "Linux",
            "^Other$": "unknown",
        }

        df["user_os"] = df["user_os"].replace("Not provided", "unknown")
        df["user_os"] = df["user_os"].replace("\\N", "unknown")
        df["user_os"] = pd.DataFrame(df["user_os"]).replace(regex=os_lookups)["user_os"]
        """
        \\N              384369
        Not provided      1045
        Linux               85
        Windows 8.1         70
        Windows 10          66
        Mac OS X            37
        Windows 7           15
        """
        if org_id == 1 or org_id is None:
            # A few not unknowns....
            cols_to_encode.append("user_os")
        else:
            # All unknowns....
            feature_drop_list.append("user_os")
        # user_os_version -- The operating version of the user that recommendations are given to
        # VOID - almost all values are \N...
        feature_drop_list.append("user_os_version")

        # user_java_version -- The Java version of the user that recommendations are given to
        # might be useful? Maybe recommendations may not work correctly if using older version?
        # VOID FAKE NEWS: only 7 non \\N values.... useless
        feature_drop_list.append("user_java_version")

        # user_timezone -- The local time zone of the user that recommendations are given to
        # VOID - almost all values are \N...
        feature_drop_list.append("user_timezone")

        # country_by_ip -- The country of the user, based on the user's IP
        # Create nans so when we use dummies, no colomn is made for nan values
        df["country_by_ip"] = df["country_by_ip"].replace("\\N", np.nan)
        cols_to_encode.append("country_by_ip")

        # timezone_by_ip -- The local time zone of the user that recommendations are given to (should be identical with user_timezone)
        # FAKE NEWS: definitely not at all similar to user_timezone
        # Looks difficult to parse, not sure if it is useful anyway, we can use country_by_ip for country
        feature_drop_list.append("timezone_by_ip")

        # local_time_of_request -- The local time of the user that recommendations are given to
        # Think local_time_of_request is simplier and contains same info. (Could extract day of week? will require some string parsing)
        df.loc[df.local_time_of_request == "\\N", "local_time_of_request"] = "unknown"
        feature_drop_list.append("local_time_of_request")

        # local_hour_of_request -- The local hour of the user that recommendations are given to
        # VOID: hour_request_received has same but more complete info
        # NOTE: setting unknowns as -1... not sure if this isbest approach
        df.loc[df.local_hour_of_request == "\\N", "local_hour_of_request"] = -1
        df["local_hour_of_request"] = pd.to_numeric(df["local_hour_of_request"])
        feature_drop_list.append("local_hour_of_request")

        # number_of_recs_in_set -- The number of recommendations in the recommendation set. This data is not available in the test set.
        # So, you can use this field to analyze and e.g. filter data, but not for training the model.
        # Cant change, not in test set - nothing interesting here anyways
        """
        7    264284
        3    100121
        5     16470
        6      1562
        2      1257
        1      1072
        4       921
        """
        feature_drop_list.append("number_of_recs_in_set")
        # algorithm_class -- Darwin & Goliath created recommendations with 5 different recommendation approaches/classes:
        # contentbasedfiltering, sentenceembeddings, stereotype, sameauthor, and random. contentbasedfiltering and sentenceembeddings
        # are rather similar and create recommendations based on the terms in the items' titles and abstracts/descriptions. 'Stereotype'
        # recommendations are items that were manually selected by the operators (e.g. MyVolts and JabRef). For instance, 'stereotype
        # recommendations' in JabRef recommend books about academic writing and research because we assume the users of JabRef will like this.
        # 'sameauthor' recommends documents authored by the first author of the source document. Random recommendations randomly select items
        # from our database.
        cols_to_encode.append("algorithm_class")
        """
        content_based_filtering    275071
        sentence_embeddings         81931
        unknown                     13026
        stereotype                  10735
        same_author                  4600
        random                        324
        """

        # recommendation_algorithm_id_used -- For the aforementioned 5 recommendation approaches, Darwin & Goliath has a total of
        # 23 variations. For instance, contentbasedfiltering can be applied to the source documents' titles only, the abstract only,
        # to the abstract and title, and so on. Also, for stereotype recommendations, we have a few slight variations.
        df.loc[
            df.recommendation_algorithm_id_used == "\\N",
            "recommendation_algorithm_id_used",
        ] = 0
        df["recommendation_algorithm_id_used"] = pd.to_numeric(
            df["recommendation_algorithm_id_used"]
        )
        cols_to_encode.append("recommendation_algorithm_id_used")

        # cbf_parser, search_title, search_keywords, and search_abstract provide more details on what fields the
        # contentbasedfiltering algorithm used.
        cols_to_encode.append("cbf_parser")
        df.loc[df.search_title == "yes", "search_title"] = 1
        df.loc[df.search_title == "no", "search_title"] = -1
        df.loc[df.search_keywords == "yes", "search_keywords"] = 1
        df.loc[df.search_keywords == "no", "search_keywords"] = -1
        df.loc[df.search_abstract == "yes", "search_abstract"] = 1
        df.loc[df.search_abstract == "no", "search_abstract"] = -1

        # time_recs_recieved -- The time when the Java Script client received recommendations. This data is only available
        # since a few weeks and not for JabRef. This data will not be available in the test set. Do not use it for training
        # but only to filter data.
        # SAME as time_recs_viewed below
        feature_drop_list.append("time_recs_recieved")

        # time_recs_displayed -- The time when recommendations were displayed on the website (should be usually identical
        # to time_recs_recieved or one second later)
        # SAME AS time_recs_viewed below
        feature_drop_list.append("time_recs_displayed")

        # time_recs_viewed -- The time the recommendations were displayed in the visible area of the screen. For instance,
        # on MyVolts, recommendations are often displayed out-of-sight at the bottom of the page. Only when a user scrolls
        # down the web page, the user will see recommendations. Again, this data will not be available in the test set. Do
        # not use it for training but only to filter data. It may well be that training your model only on 'viewed' recommendations
        # will deliver the best performance because it will strongly reduce noise (but also the amount of training data).
        # Can't include next line as testing data will not contain this feature
        feature_drop_list.append("time_recs_viewed")

        # clicks -- This is the number of total clicks for the delivered recommendation set. For instance, if a set had 7
        # recommendations, and 3 recommendations were clicked, then this field is '3' (multiple clicks on the same recommendation
        # are only counted once). This number should never be larger than number_of_recs_in_set (for a few instances it actually
        # is larger, but that must be a bug)
        # COMMENT
        # follows a pretty neat exponential decay distribution... not sure how useful
        # Could be used as a metric for how good an algo was given that it was used, i.e. given that the clicked, how much over the bar was it?
        # TESTING COMMENT
        # All nA in testing data, will drop for now
        feature_drop_list.append("clicks")

        # ctr -- The Click-Through rate of the model_argset, i.e. clicks divided by number_of_recs_in_set. This means CTR should usually
        # be between 0 and 1. However, a few rows are larger than 1. We do not know why.
        # COMMENT
        # This is interesting as the CTR appears to only have a few unique values... due to it being calculated with both numerator
        # and denominator being small (think 1/7, 2/7, 1/5 etc.)
        # TESTING COMMENT        model = sklearn.mixture.Gaussif __name__ == "__main__":
        # All nA in testing data, will drop for now
        feature_drop_list.append("ctr")

        # set_clicked -- '1' if at least one recommendation was clicked, '0' otherwise.
        # Interesting stats...
        # VALUE COUNTS
        # 0    378924
        # 1      6763

        te = stats.get(
            "target_encoder",
            TargetEncoder(
                verbose=0,
                cols=cols_to_encode,
                drop_invariant=False,
                return_df=True,
                handle_missing="value",
                handle_unknown="value",
                min_samples_leaf=1,
                smoothing=1.0,
            ),
        )

        if is_training:
            df = te.fit_transform(df, y=y)
        else:
            df = te.transform(df)

        if stats.get("prune_correlation", False):
            if is_training:
                # Do feature selection by correlation - corr() only handles numeric data
                corr_df = df.copy()
                corr_df["label"] = y
                corr = corr_df.corr()
                corr_target = abs(corr["label"])
                # cols_to_keep = list(corr_target[corr_target > 0.01].index)
                corr_drop_list = list(
                    corr_target[corr_target <= stats.get("corr_thresh", 0.01)].index
                )
                # Compare to_kep list against original drop_list
                for feat in corr_drop_list:
                    if feat not in feature_drop_list:
                        feature_drop_list.append(feat)
                stats.update({"corr_drop_list": feature_drop_list})
            else:
                feature_drop_list = stats.get("corr_drop_list")

        df.drop(columns=feature_drop_list, inplace=True)
        x = df.to_numpy(dtype=precision)

        stats.update({"target_encoder": te})

        return x, y, stats

    def process_testing(self, df, stats):
        """
        Process / manipulate the test data to shape it similarly to the training data
        :param df: DataFrame of test data
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: X (features)
        """
        processed, _, stats = self.process_training(df, is_training=False, **stats)
        return processed, stats

    @staticmethod
    def resample_train_data(x_train, y_train, over=True):
        """
        Currently testing methods or re-sampling imbalanced dataset.
        :param x_train: Training explanatory features to be re-sampled
        :param y_train: Training explained features to be re-sampled
        :param over: kwarg to oversample data
        :return: x_train_res, y_train_res (re-sampled training dataset)
        """
        if over:
            rs = BorderlineSMOTE(
                sampling_strategy="auto",
                random_state=69,
                k_neighbors=5,
                n_jobs=8,
                m_neighbors=10,
                kind="borderline-1",
            )
        else:
            rs = NeighbourhoodCleaningRule(
                sampling_strategy="auto",
                return_indices=False,
                random_state=69,
                n_neighbors=3,
                kind_sel="all",
                threshold_cleaning=0.1,
                n_jobs=8,
                ratio=None,
            )
            # rs = NearMiss(
            #     sampling_strategy="auto",
            #     return_indices=False,
            #     random_state=69,
            #     version=1,
            #     n_neighbors=3,
            #     n_neighbors_ver3=3,
            #     n_jobs=8,
            #     ratio=None,
            # )

        print("Before reSampling, the shape of train_X: {}".format(x_train.shape))
        print("Before reSampling, the shape of train_y: {} \n".format(y_train.shape))

        print("Before reSampling, counts of label '1': {}".format(sum(y_train == 1)))
        print("Before reSampling, counts of label '0': {}".format(sum(y_train == 0)))

        x_train_res, y_train_res = rs.fit_sample(x_train, y_train)

        print("After reSampling, the shape of train_X: {}".format(x_train_res.shape))
        print("After reSampling, the shape of train_y: {} \n".format(y_train_res.shape))

        print("After reSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
        print("After reSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

        return x_train_res, y_train_res

    @staticmethod
    def ready_training(x, y, split=0.2, state=42):
        """
        Perform train/validate split to provide a validation set to assess models and attempt to avoid overfitting
        :param x: Full set of features
        :param y: Full set of labels
        :param split: Fraction to split, train/validate
        :param state: Random seed for splitting
        :return: training features, validation features, training labels, validation labels
        """
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=split, random_state=state
        )

        return x_train, x_val, y_train, y_val

    @staticmethod
    def train(model, x, y, **kwargs):
        """
        Perform model training
        :param model: Model to be trained
        :param x: Features
        :param y: Labels
        :param kwargs: Optional kwargs to be used for training, allows for extension of function
        :return: trained model
        """
        print("Training")
        start = timer()
        trained_model = model.fit(x, y=y)

        end = timer()
        dur = end - start
        print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))

        return trained_model

    def train_eval(self, model, x, y=None, best_model=True):
        """
        Perform model training with a validation set
        :param model: Model to be trained
        :param x: Full set of features
        :param y: Full set of labels
        :param best_model: Boolean, to save best model epoch based on validation
        :return: trained model
        """
        print("Training")
        x_train, x_val, y_train, y_val = self.ready_training(x, y)

        start = timer()
        trained_model = model.fit(
            x_train, y=y_train, eval_set=(x_val, y_val), use_best_model=best_model
        )

        end = timer()
        dur = end - start
        print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))

        return trained_model

    @staticmethod
    def predict(model, x, **d):
        """
        Helper function to allow extension and overrides when using model to predict labels
        :param model: Model for prediction
        :param x: features
        :param d: kwargs, allowing for extension of function via overrides
        :return: model predictions
        """
        return model.predict(x)

    @staticmethod
    def model_stats(y, y_pred, regression=False):
        """
        Record prediction statistics of model using training data
        :param y: ground truth labels
        :param y_pred: predicted labels
        :param regression: Boolean, if regression was the method of prediction
        :return:
        """
        if len(np.unique(y_pred)) == 1:
            print("WARNING: Predicting all 0s")
        else:
            print_df_stats(y_pred)
        if regression:
            # RMSE - Root Mean Squared Error
            rmse = sqrt(sklearn.metrics.mean_squared_error(y, y_pred))
            print(f"Training RMSE Score: {rmse:,.5f}")
            # MAE - Mean Absolute Error
            mae = sklearn.metrics.mean_absolute_error(y, y_pred)
            print(f"Training MAE Score: {mae:,.5f}")
        else:
            f1_score = sklearn.metrics.f1_score(y, y_pred, average="weighted")
            print(f"Training F1 Score: {f1_score}")
            mcc = sklearn.metrics.matthews_corrcoef(y, y_pred)
            print(f"Training Matthews Correlation Coefficient: {mcc}")


class MultiModelClass(ModelClass):
    """
    Inherits ModelClass, provides framework for splitting dataset up into arbitrary components
    so multiple models can be fit to each one at a time.
    """

    def __init__(self, training=False, is_base=False):
        if is_base:
            super().__init__(training=training)
        self.name += "_Multi"
        self.test_matches = dict()

    def process_training(
        self, df, is_training=True, is_regression=False, other_df=None, **stats
    ):
        """
        Converts training data into separate website datasets, so a model can be built
        for each individually. Then invokes process training generally for each dataset.
        :param df: data DataFrame
        :param is_training: true if splitting dataset into train test
        :param is_regression: if using click through (CTR) regression approach
        :param other_df: Used for testing dataset for creating count features
        :param stats: useful statistics
        :return: x_dict, y_dict, stats_dict
        """
        orgs = df["organization_id"].unique()
        print(df["organization_id"].unique())
        x_dict = dict()
        y_dict = dict()
        stats_dict = dict()
        df["index"] = np.arange(len(df))
        for org in orgs:
            org_df = pd.DataFrame(df[df["organization_id"] == org])
            org_df = org_df.drop(["index"], axis=1)
            org_other_df = (
                pd.DataFrame(other_df[other_df["organization_id"] == org])
                if type(other_df) is pd.DataFrame
                else None
            )
            res = super().process_training(
                org_df,
                is_training=is_training,
                is_regression=is_regression,
                other_df=org_other_df,
                org_id=org,
                **stats,
            )
            x_dict.update({org: res[0]})
            y_dict.update({org: res[1]})
            stats_dict.update({org: res[2]})

        return x_dict, y_dict, stats_dict

    def process_testing(self, df, stats):
        """
        Process / manipulate the test data to shape it similarly to the training data
        :param df: DataFrame of test data
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: X (features)
        """
        orgs = df["organization_id"].unique()
        processed_dict = dict()
        stats_dict = dict()
        df["index"] = np.arange(len(df))
        other_df = stats.get("other_df")
        for org in orgs:
            org_df = pd.DataFrame(df[df["organization_id"] == org])
            self.test_matches.update({org: org_df["index"].to_numpy(dtype=int)})
            org_df = org_df.drop(["index"], axis=1)
            org_other_df = (
                pd.DataFrame(other_df[other_df["organization_id"] == org])
                if type(other_df) is pd.DataFrame
                else None
            )
            res = super().process_training(
                org_df,
                is_training=False,
                other_df=org_other_df,
                org_id=org,
                **stats[org],
            )
            processed_dict.update({org: res[0]})
            stats_dict.update({org: res[2]})
        return processed_dict, stats_dict

    def ready_training(self, x, y, split=0.2, state=42):
        """
        Splits dataset into train and test set, for each of the individual websites.
        :param x: dict of training x values ready to be split.
        :param y: dict of training y values ready to be split.
        :param split: proportion of data to be used as training data.
        :param state: Random state value for test split.
        :return: x_train_dict, x_val_dict, y_train_dict, y_val_dict
        """
        x_train_dict = dict()
        x_val_dict = dict()
        y_train_dict = dict()
        y_val_dict = dict()
        try:
            for key in x.keys():
                x_train, x_val, y_train, y_val = super().ready_training(
                    x[key], y[key], split=split, state=state
                )
                x_train_dict.update({key: x_train})
                x_val_dict.update({key: x_val})
                y_train_dict.update({key: y_train})
                y_val_dict.update({key: y_val})
        except AttributeError:
            return super().ready_training(x, y, split=split, state=state)

        return x_train_dict, x_val_dict, y_train_dict, y_val_dict

    def resample_train_data(self, x_train, y_train, over=True):
        """
        Currently testing methods or re-sampling imbalanced dataset.
        :param x_train: Training explanatory features to be re-sampled
        :param y_train: Training explained features to be re-sampled
        :param over: kwarg to oversample data
        :return: x_train_res, y_train_res (re-sampled training dataset)
        """
        x_train_res_dict = dict()
        y_train_res_dict = dict()

        for key in x_train.keys():
            x_train_res, y_train_res = super().resample_train_data(
                x_train[key], y_train[key], over=over
            )
            x_train_res_dict.update({key: x_train_res})
            y_train_res_dict.update({key: y_train_res})

        return x_train_res_dict, y_train_res_dict

    def train(self, model, x, y):
        """
        Perform model training, building model instances as needed
        :param model: Model to be trained
        :param x: Features
        :param y: Labels
        :return: trained model
        """
        trained_model_dict = dict()
        for key in x.keys():
            model = self.build()
            trained_model = super().train(model, x[key], y[key])
            trained_model_dict.update({key: trained_model})

        return trained_model_dict

    def train_eval(self, model, x, y):
        """
        Perform model training with a validation set, building model instances as needed
        :param model: Model to be trained
        :param x: Full set of features
        :param y: Full set of labels
        :return: trained model
        """
        trained_model_dict = dict()
        for key in x.keys():
            model = self.build()
            trained_model = super().train_eval(model, x[key], y[key])
            trained_model_dict.update({key: trained_model})

        return trained_model_dict

    def predict(self, model, x, is_training=True):
        """
        Helper function to allow extension and overrides when using model to predict labels
        :param model: Model for prediction
        :param x: features
        :param is_training: if the predictions are from the training set, keep separate for detailed results, saving as dictionary.
                            if from the test set, combine predictions to match up with expected output
        :return: model predictions
        """
        if is_training:
            preds = dict()
            for key in x.keys():
                y_pred = super().predict(model[key], x[key])
                preds.update({key: y_pred})
        else:
            df_len = 0
            for key in x.keys():
                df_len += len(x[key])
            preds = np.zeros([df_len])
            for key in x.keys():
                y_pred = super().predict(model[key], x[key])
                preds[self.test_matches[key]] = y_pred

        return preds

    def model_stats(self, y, y_pred, regression=False):
        """
        Record prediction statistics of model using training data, giving detailed results from each model,
        then combined results
        :param y: ground truth labels
        :param y_pred: predicted labels
        :param regression: Boolean, if regression was the method of prediction
        :return:
        """
        y_full = np.empty([0])
        y_pred_full = np.empty([0])
        for key in y.keys():
            print(f"Key: {key}")
            super().model_stats(y[key], y_pred[key], regression=regression)
            y_full = np.concatenate((y_full, y[key]), axis=0)
            y_pred_full = np.concatenate((y_pred_full, y_pred[key]), axis=0)

        print("Full")
        super().model_stats(y_full, y_pred_full, regression=regression)


# Method outside of classes for convenience
def print_df_stats(predictions):
    """
    Helper function to print out statistics from a dataframe
    :param predictions: Input np.array of predictions to give statistics on
    """
    print("Prediction stats:")
    print(pd.DataFrame(predictions).describe())
    print(pd.Series(predictions).value_counts())


class GMM(ModelClass):
    """
    Gaussian Mixture Model (GMM) Class
    """

    def __init__(self, training):
        name = "Gaussian_Mixture_Model"
        super().__init__(name, training)  # Is equivalent to super(GMM, self).__init__()

    @staticmethod
    def build():
        """
        Specifications to build GMM Model
        :return: instance of model class
        """
        model = sklearn.mixture.GaussianMixture(
            n_components=2,
            covariance_type="full",
            tol=1e-3,
            reg_covar=1e-5,
            max_iter=1000,
            n_init=50,
            init_params="kmeans",
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=1,
            warm_start=False,
            verbose=5,
            verbose_interval=10,
        )

        return model


class GMMMulti(MultiModelClass, GMM):
    """
    Extension of GMM to provide a single GMM Model for each recommendation provider
    """

    def __init__(self, training):
        GMM.__init__(self, training)
        MultiModelClass.__init__(self)

    def process_training(
        self,
        df,
        is_training=True,
        is_regression=False,
        other_df=None,
        corr_thresh=0.01,
        prune_correlation=True,
        **stats,
    ):
        """
        Override method for processing training and utilising correlation pruning for feature selection
        :param df: DataFrame, Training file
        :param is_training: Boolean, for training
        :param is_regression: Boolean, is regression being used
        :param other_df: DataFrame, Test file - tp provide statistics
        :param corr_thresh: Threshold for feature correlation pruning
        :param prune_correlation: Boolean, to enable correlation pruning
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: (ndarray) pruned features, (ndarray) label, (dict) dataset statistics
        """
        stats.update({"prune_correlation": prune_correlation})
        stats.update({"corr_thresh": corr_thresh})
        x, y, stats = super().process_training(
            df,
            is_training=is_training,
            is_regression=is_regression,
            other_df=other_df,
            **stats,
        )

        return x, y, stats


class RFC(ModelClass):
    """
    RandomForrest Classifier (RFC) Model Class
    """

    def __init__(self, training):
        name = "Random_Forrest_Classifier"
        super().__init__(name, training)

    @staticmethod
    def build():
        """
        Specifications to build RFC Model
        :return: instance of model class
        """
        model = RandomForestClassifier(
            n_estimators=1000,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=None,
            verbose=1,
            warm_start=False,
            class_weight=None,
        )

        return model


class RFCMulti(MultiModelClass, RFC):
    """
    Extension of RFC to provide a single RFC Model for each recommendation provider
    """

    def __init__(self, training):
        RFC.__init__(self, training)
        MultiModelClass.__init__(self)

    def process_training(
        self,
        df,
        is_training=True,
        is_regression=False,
        other_df=None,
        corr_thresh=0.01,
        prune_correlation=True,
        **stats,
    ):
        """
        Override method for processing training and utilising correlation pruning for feature selection
        :param df: DataFrame, Training file
        :param is_training: Boolean, for training
        :param is_regression: Boolean, is regression being used
        :param other_df: DataFrame, Test file - tp provide statistics
        :param corr_thresh: Threshold for feature correlation pruning
        :param prune_correlation: Boolean, to enable correlation pruning
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: (ndarray) pruned features, (ndarray) label, (dict) dataset statistics
        """
        stats.update({"prune_correlation": prune_correlation})
        stats.update({"corr_thresh": corr_thresh})
        x, y, stats = super().process_training(
            df,
            is_training=is_training,
            is_regression=is_regression,
            other_df=other_df,
            **stats,
        )

        return x, y, stats


class CBC(ModelClass):
    """
    CatBoost Classifier (CBC) Model Class
    """

    def __init__(self, training):
        name = "CatBoost_Classifier"
        super().__init__(name, training)
        self.best_model = True

    def build(self):
        """
        Specifications to build CBR Model
        :return: instance of model class
        """
        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=None,
            random_seed=42,
            task_type="GPU",
            loss_function="Logloss",
            use_best_model=self.best_model,
            eval_metric="Logloss",
            name=self.name,
            l2_leaf_reg=0.9,  # default 3.0
            verbose=False,
        )

        return model

    def train(self, model, x, y):
        """
        Override method to train with a validation set
        :param model: CBR Model, to be trained
        :param x: features
        :param y: label
        :return: Trained model
        """
        return super().train_eval(model, x, y)


class CBCMulti(MultiModelClass, CBC):
    """
    Extension of CBC to provide a single CBC Model for each recommendation provider
    """

    def __init__(self, training):
        # Not using super() to be explicit
        CBC.__init__(self, training)
        MultiModelClass.__init__(self)

    def process_training(
        self,
        df,
        is_training=True,
        is_regression=False,
        other_df=None,
        corr_thresh=0.01,
        prune_correlation=False,
        **stats,
    ):
        """
        Override method for processing training and utilising correlation pruning for feature selection
        :param df: DataFrame, Training file
        :param is_training: Boolean, for training
        :param is_regression: Boolean, is regression being used
        :param other_df: DataFrame, Test file - tp provide statistics
        :param corr_thresh: Threshold for feature correlation pruning
        :param prune_correlation: Boolean, to enable correlation pruning
        :param stats: a dictionary with statistics about the training set that can be applied to the test set
        :return: (ndarray) pruned features, (ndarray) label, (dict) dataset statistics
        """
        stats.update({"prune_correlation": prune_correlation})
        stats.update({"corr_thresh": corr_thresh})
        x, y, stats = super().process_training(
            df,
            is_training=is_training,
            is_regression=is_regression,
            other_df=other_df,
            **stats,
        )

        return x, y, stats


class CBR(ModelClass):
    """
    CatBoost Regressor (CBR) Model Class
    """

    def __init__(self, training):
        name = "CatBoost_Regressor"
        super().__init__(name, training)
        self.best_model = True
        self.regression = True

    def build(self):
        """
        Specifications to build CBR Model
        :return: instance of model class
        """
        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=None,
            random_seed=42,
            use_best_model=self.best_model,
            task_type="GPU",
            loss_function="RMSE",
            name=self.name,
            eval_metric="MAE",
            l2_leaf_reg=0.2,  # default 3.0
            verbose=False,
        )

        return model

    def train(self, model, x, y):
        """
        Override method to train with a validation set
        :param model: CBR Model, to be trained
        :param x: features
        :param y: label
        :return: Trained model
        """
        return super().train_eval(model, x, y)


class CBRMulti(MultiModelClass, CBR):
    """
    Extension of CBR to provide a single CBR Model for each recommendation provider
    """

    def __init__(self, training):
        CBR.__init__(self, training)
        MultiModelClass.__init__(self)
