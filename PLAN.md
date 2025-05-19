# Project Plan

This document outlines the planned improvements for the aws-mlops-start-kit project.

## Goals
- Provide an extendable template for building end-to-end MLOps workflows on AWS SageMaker.
- Offer configuration options for different environments (development, staging, production).
- Include automated testing and CI/CD integrations.
- Expand example datasets and models.
- Add monitoring and security best practices.

## Initial Steps
1. Introduce environment-specific configuration files under `conf/environments/`.
2. Allow the pipeline launcher to load the desired configuration via an environment variable.
3. Document how to run the pipeline with a custom environment config.

Further tasks such as additional pipeline steps, tests, and deployment automation will be iteratively added in future updates.
