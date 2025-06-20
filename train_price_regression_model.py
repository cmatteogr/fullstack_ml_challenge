"""

"""
from pipeline.evaluation import evaluation
from pipeline.preprocess import preprocess
from pipeline.train import train


def main(dataset_filepath, results_folder: str, test_size: float = 0.2):
    # read dataset
    X_train, X_test, y_train, y_test = preprocess(dataset_filepath, results_folder, test_size)

    train_results_dict, model_filepath = train(X_train, y_train, results_folder)

    results_dict = evaluation(X_test, y_test, model_filepath)


if __name__ == "__main__":
    dataset_filepath = './data/train.csv'
    results_folder = './results'
    main(dataset_filepath, results_folder)


