# A Conceptual Guide to Causal Inference in this Project

This guide explains the key causal inference concepts demonstrated in this project. The goal is to provide an intuitive understanding of why some statistical methods fail and others succeed in estimating the true causal effect of a treatment.

## The Goal: Estimating Causal Effects

In this project, we want to determine the true causal effect of a drug on the risk of an adverse drug reaction (ADR). A simple comparison of outcomes between treated and untreated groups is often misleading because the groups may not be comparable. This is where causal inference methods become essential.

## Key Concepts

### 1. Confounding

A **confounder** is a variable that is associated with both the treatment and the outcome, creating a "back-door" path that can bias the estimated effect.

*   **In our simulation:** `disease severity` is a classic confounder.
    *   Patients with more severe disease are more likely to receive the drug (treatment).
    *   Patients with more severe disease are also more likely to experience an adverse reaction (outcome), regardless of the drug.

If we don't account for disease severity, we might wrongly attribute the higher rate of ADRs in the treated group to the drug itself, when it's actually due to the underlying severity. This is why the **Naive Estimation** fails—it ignores confounders.

**Solution:** To block the back-door path, we must **adjust** for the confounder. This can be done by including it in a regression model (like in the "Correct MLE" approach) or by using Inverse Probability Weighting (IPW).

### 2. Collider Bias

A **collider** is a variable that is caused by both the treatment and the outcome. Adjusting for a collider is a common mistake that can introduce bias, even when none existed before.

*   **In our simulation:** `hospitalization` is a collider.
    *   The drug (treatment) might cause side effects that lead to hospitalization.
    *   The adverse reaction (outcome) itself can also lead to hospitalization.

When we adjust for `hospitalization` (e.g., by including it in a regression model), we are effectively looking at a very specific subset of the population. Within this subgroup, a spurious association between the drug and the ADR can be created. This is why the **Biased MLE (Collider-Adjusted)** method fails. It artificially creates a correlation that does not exist in the general population.

**Solution:** The solution is simple: **do not** adjust for colliders.

### 3. Inverse Probability Weighting (IPW)

IPW is a technique used to correct for confounding bias. It works by creating a "pseudo-population" in which the confounders are no longer associated with the treatment.

Here's how it works:
1.  **Fit a Propensity Score Model:** We model the probability of receiving the treatment, given the confounders. This probability is called the **propensity score**.
2.  **Calculate Weights:** Each individual is given a weight that is the inverse of their propensity score.
    *   Individuals who received the treatment but had a low probability of getting it (e.g., a healthy person who took the drug) get a high weight.
    *   Individuals who did not receive the treatment but had a high probability of getting it (e.g., a very sick person who did not take the drug) also get a high weight.
3.  **Apply Weights:** We then run our analysis on this weighted population. The weights balance the confounders across the treatment and control groups, mimicking a randomized controlled trial.

The **Correct IPW** method in our analysis uses this technique, successfully removing confounding bias and estimating the true causal effect.

### 4. Marginal Structural Models (MSMs)

A Marginal Structural Model (MSM) is a type of model for the outcome that is used in conjunction with IPW. Instead of directly modeling the outcome based on the treatment and confounders, an MSM models the outcome based only on the treatment in the weighted pseudo-population.

In our `causal_estimation.py` script, the `SemiMechanisticLogisticMSM` is an MSM. The `EstimatingEquationSolver` then finds the parameters of this model that best fit the data, using the IPW weights to account for confounding.

## Conclusion

This project demonstrates a critical principle of causal inference:

> The correctness of a causal estimate depends on the **causal model** you assume, not just the statistical method you use.

Both Maximum Likelihood Estimation (MLE) and IPW with Estimating Equations are powerful statistical tools. However, they only provide the right answer when we correctly identify and adjust for confounders while carefully avoiding adjustment for colliders.