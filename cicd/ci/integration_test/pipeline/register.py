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
    model_metrics = ModelMetrics(  # 定義模型應該被記錄的 Metrics
        model_statistics=MetricsSource(  # 封裝 Evaluate 的 metrics 資訊
            s3_uri=evaluate_result_uri,
            content_type="application/json",  # 定義內容格式 #TODO 訂定 evaluate result 的格式標準，藉此標準化流程
        )
    )
    image_uri = sagemaker.image_uris.retrieve(  # 指定環境鏡像檔案配置
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=model_instance["type"],
    )
    train_estimator = Estimator(  # 指定訓練用的 Estimator
        image_uri=image_uri,
        instance_type=model_instance["type"],
        instance_count=model_instance["count"],
        output_path=model_uri,
        role=role,
    )
    return RegisterModel(  # 封裝註冊模型相關資訊
        name="AbaloneRegisterModel",
        estimator=train_estimator,  # 模型訓練的 Estimator
        model_data=model_uri,
        content_types=["text/csv"],  # TODO 尚未抽象
        response_types=["text/csv"],  # TODO 尚未抽象
        inference_instances=[
            "ml.t2.medium",
            "ml.m5.xlarge",
        ],  # realtime inference AWS 機器定義
        transform_instances=["ml.m5.xlarge"],  # batch transform AWS 機器定義
        model_package_group_name=register_info["group_name"],  # 最初定義的 model group name
        approval_status=model_approval_status,  # 包裝 pipeline property 在這使用(??)
        model_metrics=model_metrics,  # metrics
    )
