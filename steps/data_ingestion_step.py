import os

import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    ingestor = DataIngestorFactory.get_data_ingestor(os.path.splitext(file_path)[1])
    return ingestor.ingest(file_path)
