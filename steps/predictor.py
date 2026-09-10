import json
import numpy as np
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string (orient="split").

    Returns:
        np.ndarray: The model's prediction.
    """
    service.start(timeout=10)
    data = json.loads(input_data)
    data.pop("columns", None)
    data.pop("index", None)
    return service.predict(np.array(data["data"]))
