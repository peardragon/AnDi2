# AnDi2 Challenge Code
This repository contains the code and models for the AnDi2 Challenge.

## Included Models
Five pre-trained models are included in this repository.

## Running the Evaluation
To evaluate the models on a dataset, follow these steps:

- Ensure the dataset file (public_data_challenge_v0.zip) is in the same directory as the evaluation script (dataset_evaluation_pipeline.ipynb).
- Run the dataset_evaluation_pipeline.ipynb notebook. This will generate the results for the dataset using the provided method.
- Upon successful completion, the results will be saved as "res.zip" in the "challenge_results" folder.

### Notes
The evaluation process is performed on a trajectory-by-trajectory basis using CPU computation.
Due to CPU computation (not parallel), the total elapsed time for the evaluation can be approximately one hour.
