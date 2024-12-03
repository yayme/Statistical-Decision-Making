# Statistical Decision Making

This repository includes exploratory projects focused on statistical decision-making techniques, providing hands-on applications of key methods in various contexts, including healthcare prediction, inventory management, and Bayesian inference.

## Projects

### 1. Bayesian Inference
This project applies **Bayesian inference** to update beliefs about uncertain parameters. Using a **Beta distribution** as a conjugate prior for a binomial likelihood, the project calculates the posterior distribution and performs **posterior predictions**. The key concepts include:
- **Beta distribution**: Used as a prior in Bayesian updating.
- **Posterior analysis**: Updating prior knowledge with observed data to refine decisions.
- **Credible intervals**: Estimating uncertainty around predictions.

### 2. Newsvendor Problem
This project addresses the **Newsvendor Problem**, a typical inventory management problem involving demand uncertainty. The project covers:
- **Censored demand**: When demand is partially observed, requiring techniques to handle missing or incomplete data.
- **Uncensored demand**: Full demand data is used to estimate optimal inventory levels.
- **Weather data integration**: Incorporating external data (weather) to predict demand and adjust purchasing decisions accordingly.

Files include:
- `Notebook.ipynb`: Jupyter Notebook explaining the approach.
- `sales.csv`: Sales data for demand estimation.
- `MLE Newsvendor Problem.png`: Graphical visualization of the solution.

### 3. Readmission Prediction to Clinic
This project implements **classification models** for predicting patient readmissions. Models such as **Logistic Regression** and **K-Nearest Neighbors (KNN)** are used, where the predicted probabilities are compared to a **threshold** to decide on the readmission risk. The threshold is optimized to minimize **misclassification costs**, where the cost criteria reflect the clinical consequences of false positives and negatives.

Files include:
- `Readmission_Prediction_for_Healthcare.ipynb`: Notebook detailing model implementation and threshold optimization.
- `readmission.csv`: Dataset containing healthcare information.

## Languages Used
- Python 3.x

## Libraries Used
- `numpy` for numerical computations.
- `pandas` for data manipulation and analysis.
- `scipy` for statistical methods.
- `matplotlib`, `seaborn` for data visualization.
- `sklearn` for machine learning models and evaluation.
- `jupyter` for interactive notebooks.


