import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep


def get_training_step(
    region, role, model_path, train_data, validation_data, model_info, instance
):
    image_uri = sagemaker.image_uris.retrieve(  # Retrieve the training image
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=instance["type"],
    )
    train_estimator = Estimator(  # Estimator used for training
        image_uri=image_uri,
        instance_type=instance["type"],
        instance_count=instance["count"],
        output_path=model_path,
        role=role,
    )
    train_estimator.set_hyperparameters(  # Set training hyperparameters
        **model_info["hyper_params"]
    )
    return TrainingStep(  # Define the training step
        name="AbaloneTrain",
        estimator=train_estimator,
        inputs={"train": train_data, "validation": validation_data},
    )
