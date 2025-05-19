from datetime import datetime

import sagemaker
from sagemaker.model import Model


def deploy_model(model_uri, role, region, model_info, instance_type, sagemaker_session):
    """Deploy the trained model to a SageMaker endpoint."""
    image_uri = sagemaker.image_uris.retrieve(
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=instance_type,
    )
    model = Model(
        image_uri=image_uri,
        model_data=model_uri,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    endpoint_name = f"DEMO-{datetime.utcnow():%Y-%m-%d-%H%M}"
    print("EndpointName =", endpoint_name)
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    return endpoint_name
