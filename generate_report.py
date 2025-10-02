"""
Generates a visual report comparing different causal estimation strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from causal_estimation import EstimatingEquationSolver, SemiMechanisticLogisticMSM, fit_propensity_score, compute_ipw
from data.simulation import simulate_adr_data

def generate_visual_report():
    """
    Runs the simulation, performs estimations, and generates a comparative plot.
    """
    # 1. Simulate data
    df = simulate_adr_data(n_patients=2000, seed=42)
    Y = df['Y'].values
    A = df['A'].values
    theta_true_adr = df['theta_true_adr'].iloc[0]

    # --- Result storage ---
    results = {}

    # 2. Perform Naive Estimation (EE-based)
    naive_model = SemiMechanisticLogisticMSM(Y, A)
    naive_solver = EstimatingEquationSolver(naive_model)
    theta_naive, _ = naive_solver.solve([-1, 0.5], max_iter=10, tol=1e-4) # Suppress verbose output
    ci_low_naive, ci_high_naive = naive_solver.confint(theta_naive)
    results['Naive (EE)'] = (theta_naive[1], ci_low_naive[1], ci_high_naive[1])

    # 3. Perform Standard MLE (Logistic Regression) - Adjusting for everything
    # This is standard practice in prediction, but induces collider bias for causal inference
    mle_vars = df[['A', 'L', 'age', 'sex', 'comorbidities', 'hospitalized']]
    mle_vars = sm.add_constant(mle_vars)
    logit_mle = sm.Logit(Y, mle_vars).fit(disp=0)
    mle_est = logit_mle.params['A']
    mle_ci = logit_mle.conf_int().loc['A'].values
    results['Standard MLE (Biased)'] = (mle_est, mle_ci[0], mle_ci[1])

    # 4. Perform IPW with Collider (Incorrect Adjustment)
    confounders_and_collider = df[['L', 'age', 'sex', 'comorbidities', 'hospitalized']].values
    ps_model_collider = fit_propensity_score(confounders_and_collider, A)
    ps_collider = ps_model_collider.predict_proba(confounders_and_collider)[:, 1]
    weights_collider = compute_ipw(A, ps_collider)
    ipw_model_collider = SemiMechanisticLogisticMSM(Y, A, weights=weights_collider)
    ipw_solver_collider = EstimatingEquationSolver(ipw_model_collider)
    theta_ipw_collider, _ = ipw_solver_collider.solve([-1, 0.5], max_iter=10, tol=1e-4)
    ci_low_collider, ci_high_collider = ipw_solver_collider.confint(theta_ipw_collider)
    results['Collider-Adjusted IPW'] = (theta_ipw_collider[1], ci_low_collider[1], ci_high_collider[1])

    # 5. Perform Correct IPW Estimation
    confounders_only = df[['L', 'age', 'sex', 'comorbidities']].values
    ps_model_correct = fit_propensity_score(confounders_only, A)
    ps_correct = ps_model_correct.predict_proba(confounders_only)[:, 1]
    weights_correct = compute_ipw(A, ps_correct)
    ipw_model_correct = SemiMechanisticLogisticMSM(Y, A, weights=weights_correct)
    ipw_solver_correct = EstimatingEquationSolver(ipw_model_correct)
    theta_ipw_correct, _ = ipw_solver_correct.solve([-1, 0.5], max_iter=10, tol=1e-4)
    ci_low_correct, ci_high_correct = ipw_solver_correct.confint(theta_ipw_correct)
    results['Correct IPW'] = (theta_ipw_correct[1], ci_low_correct[1], ci_high_correct[1])

    # 6. Generate the plot
    labels = list(results.keys())
    estimates = [res[0] for res in results.values()]
    lower_bounds = [res[1] for res in results.values()]
    upper_bounds = [res[2] for res in results.values()]
    errors = [np.array(estimates) - np.array(lower_bounds), np.array(upper_bounds) - np.array(estimates)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot estimates with error bars
    ax.errorbar(labels, estimates, yerr=errors, fmt='o', color='black',
                ecolor='gray', elinewidth=3, capsize=5, label='Estimated log(OR)')

    # Plot true value
    ax.axhline(y=theta_true_adr, color='red', linestyle='--', linewidth=2, label=f'True Effect = {theta_true_adr:.2f}')

    ax.set_ylabel('Causal log(OR) Estimate', fontsize=12)
    ax.set_title('Comparison of Causal Estimation Strategies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(axis='x', labelsize=12, rotation=15)

    fig.tight_layout()
    plt.savefig('causal_estimation_report.png', dpi=300)

if __name__ == "__main__":
    generate_visual_report()
    print("Report generated successfully as causal_estimation_report.png")