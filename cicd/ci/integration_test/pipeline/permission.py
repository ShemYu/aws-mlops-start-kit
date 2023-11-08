import boto3

assumed_role_name = "AssumedSageMakerExecutionRole"


def get_sagemaker_execution_role(role):
    # 创建一个 STS 客户端
    sts_client = boto3.client("sts")

    # 假设角色并获取临时凭证
    assumed_role = sts_client.assume_role(
        RoleArn=role, RoleSessionName=assumed_role_name  # 替换为您的角色 ARN  # 任意角色会话名称
    )

    # 从返回的凭证中提取访问密钥、秘密访问密钥和会话令牌
    credentials = assumed_role["Credentials"]
    return credentials
