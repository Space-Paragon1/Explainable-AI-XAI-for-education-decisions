# Explainable AI for Educational Decision-Making (XAI-ED)

## Overview
This project explores how explainable artificial intelligence (XAI) can be used
to support transparent and accountable decision-making in educational systems.
We focus on student performance prediction and generate human-interpretable
explanations for model decisions.

## Motivation
AI models are increasingly used to predict learning outcomes and personalize
education, yet most operate as black boxes. In educational contexts, opaque
decisions raise ethical, pedagogical, and trust concerns. This project aims to
bridge that gap by integrating explainability directly into educational ML systems.

## Phase 1 Objectives
- Build baseline student performance prediction models
- Compare interpretable and black-box models
- Apply SHAP to explain individual predictions
- Establish a modular research pipeline

## Models
- Logistic Regression (baseline, interpretable)
- Random Forest (higher capacity, black-box)

## Explainability
We use SHAP (SHapley Additive exPlanations) to identify feature contributions
to individual predictions and global model behavior.

## Project Structure
data/ # datasets
src/ # training, evaluation, explanation modules
outputs/ # metrics and explanation artifacts


## Future Work
- Counterfactual explanations
- Instructor vs student explanation views
- Integration with an offline AI tutor
- Fairness and bias analysis

THIS PROJECT IS STILL IN PROGRESS
