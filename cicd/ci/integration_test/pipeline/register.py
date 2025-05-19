import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel


def register_model(
    role,
    region,
    evaluate_result_uri,
    model_info,
    model_path,
    model_instance,
    model_uri,
    register_info,
    model_approval_status,
):
    model_metrics = ModelMetrics(  # Metrics recorded for the model
        model_statistics=MetricsSource(  # Wrap evaluation metrics information
            s3_uri=evaluate_result_uri,
            content_type="application/json",  # TODO define standard evaluation result format
        )
    )
    image_uri = sagemaker.image_uris.retrieve(  # Retrieve the training image
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=model_instance["type"],
    )
    train_estimator = Estimator(  # Estimator used for training
        image_uri=image_uri,
        instance_type=model_instance["type"],
        instance_count=model_instance["count"],
        output_path=model_uri,
        role=role,
    )
    return RegisterModel(  # Package registration information
        name="AbaloneRegisterModel",
        estimator=train_estimator,  # Estimator for the training job
        model_data=model_uri,
        content_types=["text/csv"],  # TODO make configurable
        response_types=["text/csv"],  # TODO make configurable
        inference_instances=[
            "ml.t2.medium",
            "ml.m5.xlarge",
        ],  # instance types for real-time inference
        transform_instances=["ml.m5.xlarge"],  # instance type for batch transform
        model_package_group_name=register_info[
            "group_name"
        ],  # initial model group name
        approval_status=model_approval_status,  # pipeline property
        model_metrics=model_metrics,  # evaluation metrics
    )
