# Statistical Decision Making

This repository includes exploratory projects focused on statistical decision-making techniques, providing hands-on applications of key methods in various contexts, including healthcare prediction, inventory management, Bayesian inference, and optimal stopping problems.

## Projects

### 1. Bayesian Inference
This project demonstrates **Bayesian inference** with a focus on parameter estimation using **Maximum Likelihood Estimation (MLE)**, **Maximum A Posteriori (MAP)**, and **posterior mean**. Given prior knowledge and observed data, the aim is to update our belief about the underlying parameters of the model.

#### Bayesian Framework:
The core idea is to compute the **posterior distribution** using Bayes' theorem:

$$
P(\theta | X) = \frac{P(X | \theta) P(\theta)}{P(X)}
$$

Where:
- \( P(\theta | X) \) is the posterior distribution of the parameter \( \theta \),
- \( P(X | \theta) \) is the likelihood function,
- \( P(\theta) \) is the prior distribution, and
- \( P(X) \) is the marginal likelihood.

In this project, we use a **Beta distribution** as the conjugate prior for a **Binomial likelihood**. The Beta distribution is given by:

$$
\text{Beta}(\theta; \alpha, \beta) = \frac{\theta^{\alpha - 1} (1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}
$$

Where \( \alpha \) and \( \beta \) are the shape parameters, and \( B(\alpha, \beta) \) is the Beta function.

#### MLE and MAP:
- **Maximum Likelihood Estimation (MLE)** is used to find the parameter \( \theta \) that maximizes the likelihood function:

$$
\hat{\theta}_{MLE} = \arg\max_{\theta} P(X | \theta)
$$

- **Maximum A Posteriori (MAP)** estimation incorporates the prior knowledge:

$$
\hat{\theta}_{MAP} = \arg\max_{\theta} P(X | \theta) P(\theta)
$$

#### Posterior Mean:
The **posterior mean** is computed as the expected value of the parameter under the posterior distribution:

$$
\mu_{\text{post}} = \int_{\theta} \theta P(\theta | X) \, d\theta
$$

This value gives the best estimate of \( \theta \) based on both the prior and observed data.

#### Loss Functions:
In decision-making, a **loss function** quantifies the cost of a decision. In this project, the **quadratic loss** function is used to penalize deviations between the estimated and true parameters:

$$
L(\theta, \hat{\theta}) = (\theta - \hat{\theta})^2
$$

Alternatively, a **logarithmic loss** could be used to measure the quality of probabilistic predictions.

---

### 2. Newsvendor Problem
This project addresses the **Newsvendor Problem**, a typical inventory management problem involving demand uncertainty. The project covers:
- **Censored demand**: When demand is partially observed, requiring techniques to handle missing or incomplete data.
- **Uncensored demand**: Full demand data is used to estimate optimal inventory levels.
- **Weather data integration**: Incorporating external data (weather) to predict demand and adjust purchasing decisions accordingly.

---

### 3. Readmission Prediction to Clinic
This project implements **classification models** for predicting patient readmissions. Models such as **Logistic Regression** and **K-Nearest Neighbors (KNN)** are used, where the predicted probabilities are compared to a **threshold** to decide on the readmission risk. The threshold is optimized to minimize **misclassification costs**, where the cost criteria reflect the clinical consequences of false positives and negatives.

---

### 4. Multiple Secretary Problem with Unknown Distribution
This project explores the **Multiple Secretary Problem**, an optimal stopping problem where decisions must be made to maximize rewards when the underlying distribution of rewards is unknown. 

#### Approach:
- **Q-Learning Table**: Reinforcement learning is applied using a tabular Q-learning approach to estimate the value of actions in each state.
- **Discounted Learning**: Future rewards are discounted to prioritize immediate gains, ensuring a balance between exploration and exploitation.
- **Epsilon-Greedy Strategy**: An adaptive epsilon value is used for exploration. Initially, the model explores more to gather information, and as learning progresses, it shifts toward exploitation by lowering epsilon.
- The algorithm learns dynamically to make decisions under uncertainty and improve the selection process over time.

---

## Languages Used
- Python 3.x

## Libraries Used
- `numpy` for numerical computations.
- `pandas` for data manipulation and analysis.
- `scipy` for statistical methods.
- `matplotlib`, `seaborn` for data visualization.
- `sklearn` for machine learning models and evaluation.
- `jupyter` for interactive notebooks.
