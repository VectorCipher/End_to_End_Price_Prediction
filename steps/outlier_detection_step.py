import logging
import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Handling outliers in DataFrame of shape: {df.shape}")
    detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    return detector.handle_outliers(df, method="cap")
