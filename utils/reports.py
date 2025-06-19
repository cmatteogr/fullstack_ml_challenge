"""
Reports utils auxiliar
"""
from ydata_profiling import ProfileReport
import pandas as pd

def compare_y_data_profiling(a_df, a_report_title, b_df, b_report_title, comparison_report_filepath):
    # Use ydata-profiling to compare input and generated data
    a_data_report = ProfileReport(a_df, title=a_report_title)
    # generate report with generated data
    b_data_report = ProfileReport(b_df, title=b_report_title)
    # compare reports, original data and generated data
    comparison_report = a_data_report.compare(b_data_report)
    # save report
    comparison_report.to_file(comparison_report_filepath)


def generate_profiling_report(title:str, report_filepath:str,  df: pd.DataFrame=None, data_filepath:str=None,
                              type_schema=None, minimal:bool=True):
    """
    Generate a profiling report for a dataframe
    :param title: report title
    :param report_filepath: report filepath
    :param df: report dataframe
    :param data_filepath: data filepath
    :param type_schema: type schema
    :param minimal: minimal profiling data to minimize data usage
    """
    # if df and data_filepath has a value then raise an Exception
    if df and data_filepath:
        raise Exception("Data should be defined from df or data_filepath, not both")

    # read data from filepath if exist
    if data_filepath:
        df = pd.read_csv(data_filepath)
    # generate data profiling report
    df_profile = ProfileReport(df, title=title, minimal=minimal, type_schema=type_schema)
    # export profiling report
    df_profile.to_file(report_filepath)