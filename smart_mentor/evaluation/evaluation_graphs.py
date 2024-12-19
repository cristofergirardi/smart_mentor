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

    columns_metrics = ["h", "model", "metric", "similarity"]
    hypotheses = ["h1", "h2", "h3", "h4", "h5", "h6"]
    metrics_name = ["bert", "codet5"]

    file_written = f"smart_mentor/resources/Metrics_median_null_hypothese.csv"
    try:
        reader.removeFile(file_written)
    except FileNotFoundError as e:
        logger.error("File does not found")
    
    df_median = pd.DataFrame(columns=columns_metrics)
    writer.write(file_written, df_median)

    for hypothese in hypotheses:
        for metric_name in metrics_name:
            df_read = reader.readFile(f"smart_mentor/resources/Metrics_{hypothese}_{metric_name}.csv")

            df_model = df_read['model'].unique()
            df_metric_type = df_read['metric'].unique()

            for mode in df_model:
                for metric in df_metric_type:
                    ## Select hypothesis and model
                    df_metrics = df_read.where(df_read['h'].str.contains(hypothese) & 
                                            df_read['model'].str.contains(mode) &
                                            df_read['metric'].str.contains(metric)
                                            ).dropna(how='all')
                    similarity = hmean(df_metrics['similarity'].apply(abs))

                    new_list = [{"h": hypothese, "model": mode, "metric": metric, "similarity": similarity}]
                    df_median = eg.add_new_row(df_median, new_list)

    ## Appending dataframe
    writer.write(file_written, df_median)            