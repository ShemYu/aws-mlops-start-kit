# aws-mlops-start-kit

This project provides a starting point for building SageMaker pipelines. It is
intended to become a **cookiecutter template** so that new MLOps projects can be
bootstrapped quickly.  The repository ships with a minimal yet complete pipeline
definition and sample scripts that you are free to replace with your own
feature engineering, training and evaluation code.

## Project structure

- `conf/` – YAML configuration files describing pipeline parameters. The default
  files `pipeline.yaml` and `pipeline2.yaml` specify input locations, estimator
  options and model registry settings.
- `cicd/ci/integration_test/pipeline/` – Python modules that assemble the
  SageMaker pipeline.  Each step (preprocess, train, evaluate, register) is kept
  in a dedicated file so you can easily modify or extend the logic.
- `src/` – Example scripts used by the pipeline. `preprocess.py` performs basic
  feature processing of the Abalone dataset and `evaluate.py` calculates metrics.
- `Makefile` – Contains a `lint` rule that formats the code using `isort` and
  `black`.

## Quick start

1. **Install dependencies** – install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the environment** – update `conf/pipeline.yaml` (or
   `conf/pipeline2.yaml`) to point to your S3 buckets, specify the training
   instance type and tweak hyper‑parameters.  The file `cicd/ut.yaml` holds the
   default bucket URI and the execution role used during development.

3. **Run the pipeline** – execute:

   ```bash
   python cicd/ci/integration_test/pipeline/main.py \
       --env-config cicd/ut.yaml \
       --pipeline-config conf/pipeline.yaml
   ```

   This creates and submits a SageMaker pipeline that includes preprocessing,
   training, evaluation and model registration steps.

4. **Deploy the model** – the helper `cicd/ci/integration_test/pipeline/deploy.py`
   demonstrates how to deploy the training output to a SageMaker endpoint.

Format the codebase at any time with `make lint`.

## CI/CD integration

The modules under `cicd/` are designed to be invoked from your CI/CD system.
By calling `python cicd/ci/integration_test/pipeline/main.py` inside a build
job you can automatically update or create the pipeline whenever changes are
pushed.  The folder structure keeps each pipeline step in its own module so that
custom logic can be dropped in without altering unrelated components.

## Extending the template

The goal is to keep the interface simple while allowing you to plug in your own
code:

- **Feature engineering** – edit `src/preprocess.py` or modify
  `cicd/ci/integration_test/pipeline/preprocess.py` to use a different processor.
- **Model training** – adjust the estimator configuration in
  `cicd/ci/integration_test/pipeline/train.py` or swap in your own training
  script.
- **Evaluation** – replace `src/evaluate.py` with custom metrics and change the
  evaluate step accordingly.
- **Additional steps** – create new modules in the pipeline folder and register
  them in `main.py`.

Feel free to reorganize the `conf/` files or pipeline modules to suit your
workflow.  The current layout aims to be small and readable so newcomers can
quickly understand where to add their own code.

