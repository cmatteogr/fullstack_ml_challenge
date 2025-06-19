# import libs
from utils.reports import generate_profiling_report
import os
import pandas as pd

# read dataset
dataset_filepath = './data/train.csv'
dataset_df = pd.read_csv(dataset_filepath)
print(dataset_df.head(10))

print(dataset_df.describe())

# generate a data profiling report using data_profiling lib
title = "Raw Dataset Profiling"
report_name = 'raw_dataset_profiling'
results_folder_path = 'results'
report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=dataset_filepath, minimal=True)