"""

"""
import xgboost as xgb
from pipeline.preprocess import map_garageSpaces, map_numOfPatioAndPorchFeatures, map_numOfBathrooms, map_numOfBedrooms
from utils.constants import REGRESSION_MODEL_FILEPATH
import pandas as pd
from sklearn.ensemble import IsolationForest
import spacy
from sklearn.preprocessing import OrdinalEncoder
from gensim.models import FastText
import numpy as np
import os
import joblib
import json
from utils.constants import ARTIFACTS_FOLDER, PREPROCESSING_ARTIFACTS_DICT_FILENAME
from utils.vectors import get_tokens_from_sentences, tokens_to_vector


def inference(dataset_filepath: str):
    # read dataset
    dataset_df = pd.read_csv(dataset_filepath)
    running_n_rows = dataset_df.shape[1]

    # read preprocess artifacts
    preprocessing_artifacts_dict_filepath = os.path.join(ARTIFACTS_FOLDER, PREPROCESSING_ARTIFACTS_DICT_FILENAME)
    with open(preprocessing_artifacts_dict_filepath, 'r') as file:
        preprocessing_artifacts = json.load(file)

    # check valid columns
    # TODO: when the finals columns were defined
    dataset_df = dataset_df[preprocessing_artifacts_dict_filepath['valid_columns']]

    # Validate inputs
    # filter them by valid values
    dataset_df = dataset_df.loc[dataset_df['city'].isin(preprocessing_artifacts['valid_cities'])]
    if running_n_rows > dataset_df.shape[1]:
        running_n_rows = dataset_df.shape[1]
        print("WARNING", "the in input has invalid cities, they will be removed")

    dataset_df = dataset_df.loc[dataset_df['homeType'].isin(preprocessing_artifacts['valid_home_types'])]
    if running_n_rows > dataset_df.shape[1]:
        running_n_rows = dataset_df.shape[1]
        print("WARNING", "the in input has invalid home types, they will be removed")

    dataset_df.pop('city')
    dataset_df.pop('homeType')

    # load models
    outliers_detection_model: IsolationForest = joblib.load(preprocessing_artifacts["outlier_removal_model_filepath"])
    features_model: FastText = joblib.load(preprocessing_artifacts["feature_w2v_filepath"])
    price_range_encoder: OrdinalEncoder = joblib.load(preprocessing_artifacts["price_range_encoder_filepath"])

    # load regression model
    regression_filepath = os.path.join(ARTIFACTS_FOLDER, REGRESSION_MODEL_FILEPATH)
    loaded_model = xgb.XGBRFRegressor()
    loaded_model.load_model(regression_filepath)

    print('remove rows with empty values')
    # remove the description empty
    dataset_df = dataset_df.loc[~dataset_df['description'].isna()]
    if running_n_rows > dataset_df.shape[1]:
        running_n_rows = dataset_df.shape[1]
        print("WARNING", "the in input has invalid empy descriptions, they will be removed")

    features_vector_size = preprocessing_artifacts["features_vector_size"]
    # apply tokenization
    nlp = spacy.load("en_core_web_sm")
    dataset_df['description_features_tokens'] = dataset_df['description_features'].map(
        lambda fi: get_tokens_from_sentences(nlp, fi))
    sentences_tokens = [item for sublist in dataset_df['description_features_tokens'] for item in sublist]
    # for each pack of sentences, get the vectors and calculate the mean
    features_s = []
    for feature_tokens in dataset_df['description_features_tokens'].tolist():
        # get vectors from sentences tokens
        sentence_vectors_pack = [tokens_to_vector(feature_single, features_model) for feature_single in
                                 feature_tokens]
        # summarize vectors in one single vector calculating the mean
        feature_summary_vector = np.mean(sentence_vectors_pack, axis=0)
        feature_summary_vector = feature_summary_vector.tolist()
        # if vector is nan then replace it by list of nan
        if type(feature_summary_vector) is not list:
            feature_summary_vector = [np.nan] * features_vector_size
        features_s.append(feature_summary_vector)

    # build feature df
    features_vectors_df = pd.DataFrame(features_s, columns=[f'feature_x{i}' for i in range(len(features_s[0]))],
                                       index=dataset_df.index)
    # fill feature nan to 0
    features_vectors_df.fillna(0, inplace=True)
    # concatenate vector representation to main dataset
    dataset_df = pd.concat([dataset_df, features_vectors_df], axis=1)
    # remove description_features and description
    dataset_df.pop('description')
    dataset_df.pop('description_features')
    dataset_df.pop('description_features_tokens')

    # apply garageSpaces transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    dataset_df['garageSpaces'] = dataset_df['garageSpaces'].apply(map_garageSpaces)

    # apply map_numOfPatioAndPorchFeatures transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    dataset_df['numOfPatioAndPorchFeatures'] = dataset_df['numOfPatioAndPorchFeatures'].apply(
        map_numOfPatioAndPorchFeatures)

    # apply lotSizeSqFt transformation
    max_lotSizeSqFt_q = preprocessing_artifacts['max_lotSizeSqFt_q']
    dataset_df = dataset_df.loc[(dataset_df['lotSizeSqFt'] < max_lotSizeSqFt_q)]
    if running_n_rows > dataset_df.shape[1]:
        running_n_rows = dataset_df.shape[1]
        print("WARNING", "the in input has invalid lotSizeSqFt, they will be removed")

    # apply map_numOfBathrooms transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    dataset_df['numOfBathrooms'] = dataset_df['numOfBathrooms'].apply(map_numOfBathrooms)

    # apply map_numOfBathrooms transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    dataset_df['numOfBedrooms'] = dataset_df['numOfBedrooms'].apply(map_numOfBedrooms)

    # Predict anomalies (-1 for outliers and 1 for inliers)
    dataset_df['outlier'] = outliers_detection_model.predict(dataset_df)
    # Remove global outliers
    dataset_df = dataset_df[dataset_df['outlier'] != -1]
    dataset_df.drop(columns='outlier', inplace=True)
    if running_n_rows > dataset_df.shape[1]:
        running_n_rows = dataset_df.shape[1]
        print("WARNING", "the in input has invalid outliers, they will be removed")

    # Use the loaded model
    y_pred = loaded_model.predict(dataset_df)

    # Inverse transform
    y_pred_category = price_range_encoder.inverse_transform(y_pred)

    # return predicted values
    return y_pred_category


if __name__ == "__main__":
    dataset_filepath = './data/train.csv'
    dataset_df = pd.DataFrame(dataset_filepath)
    dataset_df.pop('uid')

    inference_dataset_df = dataset_df.sample(100)
    inference_dataset_df.to_csv('./data/inference.csv', index=False)

    inference(inference_dataset_df)

