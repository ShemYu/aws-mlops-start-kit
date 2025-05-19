# aws-mlops-start-kit

This repository is a template for setting up end-to-end MLOps workflows on AWS SageMaker. It offers a basic pipeline definition together with sample preprocessing and evaluation scripts.

## Project structure

- `conf/` - YAML configuration files describing the pipeline. The files `pipeline.yaml` and `pipeline2.yaml` define input locations, training options and model registry settings.
- `cicd/ci/integration_test/pipeline/` - Modules that assemble each step of the SageMaker pipeline such as preprocessing, training, evaluation and registering the model.
- `src/` - Example Python scripts executed by the pipeline. `preprocess.py` processes the Abalone dataset and `evaluate.py` computes evaluation metrics.
- `Makefile` - Provides a `lint` rule which formats the source code using `isort` and `black`.

## Running the pipeline

1. Install dependencies (SageMaker SDK and scikit-learn).
2. Adjust the values in one of the configuration files under `conf/` to match your environment. You can also provide an environment file under `conf/environments/` and set the `ENV_CONFIG_PATH` environment variable to its path.
3. Launch the pipeline with:

```bash
python cicd/ci/integration_test/pipeline/main.py
```
The script will look for the file specified by `ENV_CONFIG_PATH`. If not set, it defaults to `cicd/ut.yaml`.

This command creates and submits a SageMaker pipeline containing preprocessing, training, evaluation and model registration steps.

Format the codebase at any time with:

```bash
make lint
```
