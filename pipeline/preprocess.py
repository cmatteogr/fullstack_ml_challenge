""""
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import spacy
from sklearn.preprocessing import OrdinalEncoder
from gensim.models import FastText
import numpy as np
import os
import joblib
import json
from utils.constants import ARTIFACTS_FOLDER, PREPROCESSING_ARTIFACTS_DICT_FILENAME
from utils.reports import generate_profiling_report
from utils.vectors import get_tokens_from_sentences, tokens_to_vector


def extract_features_from_desc(nlp, description: str,
                               valid_labels: list[str] = ['FAC', 'PRODUCT', 'LOC', 'ORG', 'GPE']) -> list[str]:
    # process description
    doc = nlp(description)
    if not doc.ents:
        return []
    else:
        # filter by valid values, return the list of entities
        return list(map(lambda ent: ent.text, filter(lambda ent: ent.label_ in valid_labels, doc.ents)))


def map_garageSpaces(garage_spaces: int, max_binding_value: int = 5) -> int:
    # if garage_spaces is greater than 5 then return 5
    if garage_spaces >= max_binding_value:
        return max_binding_value
    return garage_spaces


def map_numOfPatioAndPorchFeatures(num_patio_porch_features: int, max_binding_value: int = 4) -> int:
    # if garage_spaces is greater than 5 then return 5
    if num_patio_porch_features >= max_binding_value:
        return max_binding_value
    return num_patio_porch_features


def map_numOfBathrooms(num_bathrooms: int, max_binding_value: int = 6) -> int:
    # if garage_spaces is greater than 5 then return 5
    if num_bathrooms >= max_binding_value:
        return max_binding_value
    return num_bathrooms


def map_numOfBedrooms(num_bedrooms: int, max_binding_value: int = 6) -> int:
    # if garage_spaces is greater than 5 then return 5
    if num_bedrooms >= max_binding_value:
        return max_binding_value
    return num_bedrooms


def preprocess(dataset_filepath, results_folder_path, test_size: float = 0.2):
    print("start preprocessing")

    # read dataset
    dataset_df = pd.read_csv(dataset_filepath)

    print('remove irrelevant columns')
    # remove irrelevant columns
    dataset_df.pop('uid')
    valid_columns = list(dataset_df.columns)
    valid_columns =valid_columns[:-1]

    print('filter valid values')
    # filter by valid values
    # NOTE: This could change in the future with more data
    # City remove values different to austin
    valid_cities = ['austin']
    dataset_df = dataset_df.loc[dataset_df['city'].isin(valid_cities)]
    # Home type remove values different to 'Single Family'
    valid_home_types = ['Single Family']
    dataset_df = dataset_df.loc[dataset_df['homeType'].isin(valid_home_types)]

    # NOTE: in this version we can remove city, and homeType because it uses a single value
    # with more information this could change in the future
    dataset_df.pop('city')
    dataset_df.pop('homeType')

    print('remove rows with empty values')
    # remove the description empty
    dataset_df = dataset_df.loc[~dataset_df['description'].isna()]

    print('get useful data from description')
    # NOTE: the following transformation is possible before the split due the entities are calculated with a pretrained model
    # use spaCy NLP to extract features from the description
    nlp = spacy.load("en_core_web_sm")
    dataset_df['description_features'] = dataset_df[['description']].apply(
        lambda row: extract_features_from_desc(nlp, row['description']), axis=1)
    # apply word to vector to transform description_features to numbers
    features_vector_size = 2
    # apply tokenization
    dataset_df['description_features_tokens'] = dataset_df['description_features'].map(
        lambda fi: get_tokens_from_sentences(nlp, fi))
    sentences_tokens = [item for sublist in dataset_df['description_features_tokens'] for item in sublist]
    # train the FastText model
    features_model = FastText(sentences_tokens, vector_size=features_vector_size, window=5, min_count=1, sg=1)
    # for each pack of sentences, get the vectors and calculate the mean
    features_s = []
    for feature_tokens in dataset_df['description_features_tokens'].tolist():
        # get vectors from sentences tokens
        sentence_vectors_pack = [tokens_to_vector(feature_single, features_model) for feature_single in feature_tokens]
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

    feature_w2v_filepath = os.path.join(ARTIFACTS_FOLDER, 'feature_w2v_model.pkl')
    joblib.dump(features_model, feature_w2v_filepath)

    print('split dataset')
    # get target
    y = dataset_df.pop('priceRange')
    X = dataset_df

    # apply dataset split to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print('apply features transformation')
    # apply garageSpaces transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    X_train['garageSpaces'] = X_train['garageSpaces'].apply(map_garageSpaces)
    X_test['garageSpaces'] = X_test['garageSpaces'].apply(map_garageSpaces)

    # apply map_numOfPatioAndPorchFeatures transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    X_train['numOfPatioAndPorchFeatures'] = X_train['numOfPatioAndPorchFeatures'].apply(map_numOfPatioAndPorchFeatures)
    X_test['numOfPatioAndPorchFeatures'] = X_test['numOfPatioAndPorchFeatures'].apply(map_numOfPatioAndPorchFeatures)

    # apply lotSizeSqFt transformation
    max_lotSizeSqFt_q = X_train['lotSizeSqFt'].quantile(0.95)
    X_train = X_train.loc[(X_train['lotSizeSqFt'] < max_lotSizeSqFt_q)]
    X_test = X_test.loc[(X_test['lotSizeSqFt'] < max_lotSizeSqFt_q)]
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    # apply map_numOfBathrooms transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    X_train['numOfBathrooms'] = X_train['numOfBathrooms'].apply(map_numOfBathrooms)
    X_test['numOfBathrooms'] = X_test['numOfBathrooms'].apply(map_numOfBathrooms)

    # apply map_numOfBathrooms transformation
    # NOTE: this could be calculated with quantiles instead in the future, that's why it happens after the split
    X_train['numOfBedrooms'] = X_train['numOfBedrooms'].apply(map_numOfBedrooms)
    X_test['numOfBedrooms'] = X_test['numOfBedrooms'].apply(map_numOfBedrooms)

    print('apply target transformation')
    # transform the target
    price_range_order = ['0-250000', '250000-350000', '350000-450000', '450000-650000', '650000+']
    price_range_encoder = OrdinalEncoder(categories=[price_range_order])
    price_range_encoder.fit(y_train.to_frame())
    # apply to train
    y_train = price_range_encoder.transform(y_train.to_frame())
    y_train = pd.DataFrame(y_train, columns=['priceRange'], index=X_train.index)
    # apply to test
    y_test = price_range_encoder.transform(y_test.to_frame())
    y_test = pd.DataFrame(y_test, columns=['priceRange'], index=X_test.index)
    # save model
    price_range_encoder_filepath = os.path.join(results_folder_path, 'price_range_encoder.pkl')
    joblib.dump(price_range_encoder, price_range_encoder_filepath)

    print('remove outliers')
    # apply outlier removal
    iso_forest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    # Fit the model
    iso_forest.fit(X_train)
    # Predict anomalies (-1 for outliers and 1 for inliers)
    X_train['outlier'] = iso_forest.predict(X_train)
    X_test['outlier'] = iso_forest.predict(X_test)
    # Remove global outliers
    X_train = X_train[X_train['outlier'] != -1]
    y_train = y_train.loc[X_train.index]
    X_train.drop(columns='outlier', inplace=True)
    X_test = X_test[X_test['outlier'] != -1]
    y_test = y_test.loc[X_test.index]
    X_test.drop(columns='outlier', inplace=True)
    # save model
    outlier_removal_model_filepath = os.path.join(results_folder_path, 'outlier_removal_model.pkl')
    joblib.dump(price_range_encoder, outlier_removal_model_filepath)

    print('save dataset')
    # save dataset
    train_df = X_train.copy()
    train_df['priceRange'] = y_train
    train_dataset_filepath = os.path.join(results_folder_path, "train_dataset.csv")
    train_df.to_csv(train_dataset_filepath, index=False)

    print('generate reports')
    # generate output report
    title = "Preprocessed Train Dataset Profiling"
    report_name = 'preprocessed_train_dataset_profiling'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    generate_profiling_report(report_filepath=train_dataset_filepath, title=title, data_filepath=dataset_filepath,
                              minimal=False)

    # save the artifacts for inference
    # TODO
    artifacts_dict = {
        'valid_cities': valid_cities,
        'valid_home_types': valid_home_types,
        'features_vector_size': features_vector_size,
        'max_lotSizeSqFt_q': max_lotSizeSqFt_q,
        'price_range_encoder_filepath': price_range_encoder_filepath,
        'outlier_removal_model_filepath': outlier_removal_model_filepath,
        'feature_w2v_filepath': feature_w2v_filepath,
        'valid_columns': valid_columns
    }
    # Save JSON results report
    results_json_filepath = os.path.join(ARTIFACTS_FOLDER, PREPROCESSING_ARTIFACTS_DICT_FILENAME)
    with open(results_json_filepath, 'w') as f:
        json.dump(artifacts_dict, f, indent=4)

    print("start completed")

    # instead of return the dataset we could return a filepath where the dataset is saved to allow serialization, scalability
    return X_train, X_test, y_train, y_test
