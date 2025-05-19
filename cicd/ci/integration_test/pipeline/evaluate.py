import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep


def get_evaluator(
    role,
    region,
    model_info,
    train_instance,
    evaluate_instance,
    evaluate_script,
    model_uri,
    test_data_s3_uri,
):
    image_uri = sagemaker.image_uris.retrieve(  # Retrieve the training image
        framework=model_info["type"],
        region=region,
        version=model_info["version"],
        py_version="py3",
        instance_type=train_instance["type"],
    )

    script_eval = ScriptProcessor(
        image_uri=image_uri,  # Use the same image as training
        command=["python3"],
        instance_type=evaluate_instance["type"],
        instance_count=evaluate_instance["count"],
        base_job_name="script-abalone-eval",  # TODO decide on a common naming rule
        role=role,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    return ProcessingStep(  # Define the evaluation processing job
        name="AbaloneEval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/model",  # TODO unify parameter with src/evaluate.py
            ),
            ProcessingInput(
                source=test_data_s3_uri,
                destination="/opt/ml/processing/test",  # TODO unify parameter with src/evaluate.py
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),  # TODO unify parameter with src/evaluate.py
        ],
        code=evaluate_script,
        property_files=[evaluation_report],
    )
