import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep


def get_training_step(
    region, role, model_path, train_data, validation_data, model_info, instance
):
    image_uri = sagemaker.image_uris.retrieve(  # 指定環境鏡像檔案配置
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=instance["type"],
    )
    train_estimator = Estimator(  # 指定訓練用的 Estimator
        image_uri=image_uri,
        instance_type=instance["type"],
        instance_count=instance["count"],
        output_path=model_path,
        role=role,
    )
    train_estimator.set_hyperparameters(  # 設定訓練用的超參數，此處指定為固定值
        **model_info["hyper_params"]
    )
    return TrainingStep(  # 定義訓練 Step object
        name="AbaloneTrain",
        estimator=train_estimator,
        inputs={"train": train_data, "validation": validation_data},
    )
