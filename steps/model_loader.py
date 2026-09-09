from sklearn.pipeline import Pipeline
from zenml import Model, step


@step
def model_loader(model_name: str) -> Pipeline:
    # Load the model by name and version
    model = Model(name=model_name, version="production")

    # Load the pipeline artifact (assuming it was saved as "sklearn_pipeline")
    model_pipeline: Pipeline = model.load_artifact("sklearn_pipeline")

    return model_pipeline
