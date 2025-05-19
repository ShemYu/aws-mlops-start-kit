import boto3

assumed_role_name = "AssumedSageMakerExecutionRole"


def get_sagemaker_execution_role(role):
    # Create an STS client
    sts_client = boto3.client("sts")

    # Assume the role and obtain temporary credentials
    assumed_role = sts_client.assume_role(
        RoleArn=role,
        RoleSessionName=assumed_role_name,
    )

    # Extract keys and token from the returned credentials
    credentials = assumed_role["Credentials"]
    return credentials
