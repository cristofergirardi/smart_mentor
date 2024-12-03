from ..config import logging_config
from ..file.smart_writer import SmartWriter
from ..file.smart_reader import SmartReader
import pandas as pd
from scipy.stats import hmean

logger = logging_config.setup_logging()


class EvaluationGraphs():

    def add_new_row(self, df_orig : pd.DataFrame, new_rows: list):
        for new_row in new_rows:
            df_orig.loc[len(df_orig)] = new_row
        return df_orig

if __name__ == "__main__":
    reader = SmartReader()
    writer = SmartWriter()
    eg = EvaluationGraphs()
    hypothese = "h0"
    file_written = f"src/resources/Metrics_median_{hypothese}.csv"
    df_read = reader.readFile(f"src/resources/Metrics_{hypothese}.csv")

    df_median = pd.DataFrame(columns=["h", "model", "metric", "precision", "recall", "f1_score"])
    try:
        reader.removeFile(file_written)
    except FileNotFoundError as e:
        logger.error("File does not found")

    writer.write(file_written, df_median)

    df_hyp = df_read['h'].unique()
    df_model = df_read['model'].unique()
    df_metric_type = df_read['metric'].unique()

    ## Looping of each hypothesis
    for hyp in df_hyp:
        for mode in df_model:
            for metric in df_metric_type:
                ## Select hypothesis and model
                df_metrics = df_read.where(df_read['h'].str.contains(hyp) & 
                                           df_read['model'].str.contains(mode) &
                                           df_read['metric'].str.contains(metric)
                                           ).dropna(how='all')
                precision = hmean(df_metrics['precision'])
                recall = hmean(df_metrics['recall'])
                f1 = hmean(df_metrics['f1_score'])

                new_list = [{"h": hyp, "model": mode, "metric": metric, "precision": precision, "recall": recall, "f1_score": f1}]
                df_median = eg.add_new_row(df_median, new_list)

    ## Appending dataframe
    writer.write(file_written, df_median)            