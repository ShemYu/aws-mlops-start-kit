import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


def get_evaluator(role, region, model_info, train_instance, evaluate_instance, evaluate_script, model_uri, test_data_s3_uri):
    image_uri = sagemaker.image_uris.retrieve( # 指定環境鏡像檔案配置
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=train_instance["type"]
    )

    script_eval = ScriptProcessor(
        image_uri=image_uri, # 這邊沿用 Training 環境
        command=["python3"],
        instance_type=evaluate_instance["type"],
        instance_count=evaluate_instance["count"],
        base_job_name="script-abalone-eval", #TODO Naming rule TBD，就可以抽象成統一名稱
        role=role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    return ProcessingStep( # 定義 Evaluate 的 Processing Instance
        name="AbaloneEval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/model" #TODO 與 src/evaluate.py 當中參數其實應該抽象，並統一介面
            ),
            ProcessingInput(
                source=test_data_s3_uri,
                destination="/opt/ml/processing/test" #TODO 與 src/evaluate.py 當中參數其實應該抽象，並統一介面
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"), #TODO 與 src/evaluate.py 當中參數其實應該抽象，並統一介面
        ],
        code=evaluate_script,
        property_files=[evaluation_report],
    )