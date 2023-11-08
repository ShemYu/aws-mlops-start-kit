import sys
from pathlib import Path

# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取 'cicd' 目录的父目录
root_dir = current_dir.parent.parent.parent.parent
# 添加到 sys.path
sys.path.append(str(root_dir))


import boto3
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline

from cicd.ci.integration_test.pipeline import (
    evaluate,
    permission,
    preprocess,
    register,
    train,
    utils,
)

# 配置類型的參數
ut_config = utils.load_config("cicd/ut.yaml")
config = utils.load_config(ut_config["pipeline_config"])


# UT 環境參數
s3_default_bucket_uri = ut_config["s3_default_bucket_uri"]
role = ut_config["sagemaker_execution_role"]
# pipeline 參數
## General
pipeline_name = config["general"]["pipeline_name"]
## Preprocess
input_data_s3_uri = (
    s3_default_bucket_uri + config["preprocess"]["input_data"]["s3_uri_postfix"]
)
input_data_content_type = config["preprocess"]["input_data"]["content_type"]
batch_data_s3_uri = (
    s3_default_bucket_uri + config["preprocess"]["batch_data"]["s3_uri_postfix"]
)
batch_data_content_type = config["preprocess"]["batch_data"]["content_type"]
processing_instance_count = config["preprocess"]["instance"]["count"]
preprocess_py_file = str(
    root_dir / ("src/" + config["preprocess"]["instance"]["script"])
)
## Training
model_path = s3_default_bucket_uri + config["train"]["model"]["s3_uri_postfix"]
model_info = config["train"]["model"]
model_training_instance_info = config["train"]["instance"]
## Evaluate
evaluate_instance_info = config["evaluate"]["instance"]
evaluate_py_file = str(root_dir / ("src/" + evaluate_instance_info["script"]))
## Register
register_info = config["register_model"]


# Session 後配置
## AssumedSageMakerExecutionRole
credentials = permission.get_sagemaker_execution_role(role)
sagemaker_session = Session(
    boto_session=boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )
)
region = sagemaker_session._region_name
default_bucket = sagemaker_session.default_bucket()


# 抽象 pipeline 的參數介面出來
processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount", default_value=1
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)
input_data = ParameterString(
    name="InputData",
    default_value=input_data_s3_uri,
)
batch_data = ParameterString(
    name="BatchData",
    default_value=batch_data_s3_uri,
)

# Preprocessing job definition
preprocess_job = preprocess.get_preprocessing_step(
    processing_instance_count=processing_instance_count,
    role=role,
    input_data=input_data,
    py_file_path=preprocess_py_file,
)
# Training job definition
train_data = TrainingInput(
    s3_data=preprocess_job.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri,
    content_type=input_data_content_type,
)
validation_data = TrainingInput(
    s3_data=preprocess_job.properties.ProcessingOutputConfig.Outputs[
        "validation"
    ].S3Output.S3Uri,
    content_type=input_data_content_type,
)
train_job = train.get_training_step(
    region=region,
    role=role,
    model_path=model_path,
    train_data=train_data,
    validation_data=validation_data,
    model_info=model_info,
    instance=model_training_instance_info,
)
# Evaluation job definition
model_uri = train_job.properties.ModelArtifacts.S3ModelArtifacts
test_data_s3_uri = preprocess_job.properties.ProcessingOutputConfig.Outputs[
    "test"
].S3Output.S3Uri
evaluate_job = evaluate.get_evaluator(
    role=role,
    region=region,
    model_info=model_info,
    train_instance=model_training_instance_info,
    evaluate_instance=evaluate_instance_info,
    evaluate_script=evaluate_py_file,
    model_uri=model_uri,
    test_data_s3_uri=test_data_s3_uri,
)
# Model registry definition
evaluate_result_uri = "{}/evaluate.json".format(
    evaluate_job.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
)
register_job = register.register_model(
    role=role,
    region=region,
    evaluate_result_uri=evaluate_result_uri,
    model_info=model_info,
    model_path=model_path,
    model_instance=model_training_instance_info,
    model_uri=model_uri,
    register_info=register_info,
    model_approval_status=model_approval_status,
)
# Pipeline definition
pipeline = Pipeline(  # 終於我們來到 pipeline 定義
    name=pipeline_name,
    parameters=[
        processing_instance_count,
        model_approval_status,
        input_data,
        batch_data,
    ],
    steps=[preprocess_job, train_job, evaluate_job, register_job],
)
pipeline.upsert(role_arn=role)
