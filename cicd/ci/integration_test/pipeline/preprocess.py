from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

framework_version = "0.23-1"


def get_preprocessing_step(processing_instance_count, role, input_data, py_file_path):
    sklearn_processor = SKLearnProcessor(  # Use the SKLearnProcessor
        framework_version=framework_version,  # Define the sklearn version
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,  # Number of instances
        base_job_name="sklearn-abalone-process",
        role=role,
    )
    return ProcessingStep(  # Package into a processing job
        name="AbaloneProcess",
        processor=sklearn_processor,
        inputs=[  # Automatically connect the input
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[  # Automatically connect the outputs
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=py_file_path,
    )
