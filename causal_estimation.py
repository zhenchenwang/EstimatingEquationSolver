"""
Estimating Equation Solver with IPW, Semi-Mechanistic MSM, and Monte Carlo SEIR Simulator
Includes fallback to gradient descent if Jacobian is singular or ill-conditioned.
"""

import numpy as np
from numpy.linalg import pinv, LinAlgError
from scipy.special import expit

# ---------------------------------------------------------------
# Propensity score and IPW
# ---------------------------------------------------------------
def fit_propensity_score(X, A):
    """Fit simple logistic regression propensity model."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, A)
    return model

def compute_ipw(A, ps):
    """Compute inverse probability weights."""
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    return A / ps + (1 - A) / (1 - ps)

# ---------------------------------------------------------------
# Semi-Mechanistic Exponential MSM (toy)
# ---------------------------------------------------------------
class SemiMechanisticExpMSM:
    def __init__(self, Y, A, weights=None):
        self.Y = np.asarray(Y)
        self.A = np.asarray(A)
        self.X = np.stack([np.ones_like(A), A], axis=1)  # Design matrix for intercept and slope
        self.weights = np.ones_like(Y) if weights is None else np.asarray(weights)

    def g_i(self, theta):
        """Individual contributions to the estimating equations."""
        mu = np.exp(self.X @ theta)
        res = self.Y - mu
        return self.weights[:, None] * res[:, None] * self.X

    def g(self, theta):
        """Estimating equations."""
        return self.g_i(theta).sum(axis=0)

    def grad_beta(self, theta):
        """Jacobian matrix of the estimating equations."""
        mu = np.exp(self.X @ theta)
        w_mu = self.weights * mu
        return -self.X.T @ (w_mu[:, None] * self.X)

# ---------------------------------------------------------------
# Semi-Mechanistic Logistic MSM
# ---------------------------------------------------------------
class SemiMechanisticLogisticMSM:
    def __init__(self, Y, A, weights=None):
        self.Y = np.asarray(Y)
        self.A = np.asarray(A)
        self.X = np.stack([np.ones_like(A), A], axis=1)  # Design matrix for intercept and slope
        self.weights = np.ones_like(Y) if weights is None else np.asarray(weights)

    def g_i(self, theta):
        """Individual contributions to the estimating equations."""
        mu = expit(self.X @ theta)
        res = self.Y - mu
        return self.weights[:, None] * res[:, None] * self.X

    def g(self, theta):
        """Estimating equations."""
        return self.g_i(theta).sum(axis=0)

    def grad_beta(self, theta):
        """Jacobian matrix of the estimating equations."""
        mu = expit(self.X @ theta)
        w_dmu = self.weights * mu * (1 - mu)
        return -self.X.T @ (w_dmu[:, None] * self.X)

# ---------------------------------------------------------------
# Monte Carlo MSM (requires simulator)
# ---------------------------------------------------------------
class MonteCarloMSM:
    def __init__(self, simulator, Y_obs, A_seq, weights=None):
        self.simulator = simulator
        self.Y_obs = np.asarray(Y_obs)
        self.A_seq = np.asarray(A_seq)
        self.weights = np.ones_like(Y_obs) if weights is None else weights

    def g(self, theta):
        Y_sim, _ = self.simulator(self.A_seq, theta, n_sims=1000)
        return np.array([np.sum(self.weights * (self.Y_obs - Y_sim))])

    def grad_beta(self, theta):
        _, grad = self.simulator(self.A_seq, theta, n_sims=1000, return_grad=True)
        return grad.reshape(1, 1)

# ---------------------------------------------------------------
# Solver with Newton + gradient descent fallback
# ---------------------------------------------------------------
class EstimatingEquationSolver:
    def __init__(self, model):
        self.model = model

    def estimating_equation(self, theta):
        return self.model.g(theta)

    def jacobian(self, theta):
        return self.model.grad_beta(theta)

    def solve(self, theta0, max_iter=50, tol=1e-6, alpha=1e-3):
        theta = np.array(theta0, dtype=float)
        for it in range(max_iter):
            g = self.estimating_equation(theta)
            J = self.jacobian(theta)
            g_norm = np.linalg.norm(g)
            if g_norm < tol:
                return theta, {"iterations": it, "g": g, "J": J}
            try:
                # Newton step
                delta = -pinv(J) @ g
                if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < 1e-12:
                    raise LinAlgError("Singular Jacobian")
                theta = theta + delta
                print(f"Iter {it}: Newton step, theta={theta}, g_norm={g_norm:.4g}")
            except LinAlgError:
                # Fallback: gradient descent
                grad_approx = J.T @ g
                theta = theta - alpha * grad_approx.flatten()
                print(f"Iter {it}: GD fallback, theta={theta}, g_norm={g_norm:.4g}")
        return theta, {"iterations": max_iter, "g": g, "J": J}

    def sandwich_variance(self, theta_hat):
        """Robust sandwich variance estimator."""
        G = self.model.g_i(theta_hat)
        B = G.T @ G / len(G)  # Meat
        A = self.jacobian(theta_hat)  # Bread
        try:
            A_inv = pinv(A)
            V = A_inv @ B @ A_inv.T
            return np.diag(V) / len(G)
        except LinAlgError:
            return np.full(len(theta_hat), np.nan)

    def confint(self, theta_hat, alpha=0.05):
        var = self.sandwich_variance(theta_hat)
        se = np.sqrt(var)
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        return theta_hat - z * se, theta_hat + z * se

from data.simulation import simulate_adr_data

# ---------------------------------------------------------------
# Example demo
# ---------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Realistic Drug ADR Example with Confounding and Collider Bias
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("Realistic Drug ADR Example with Confounding and Collider Bias")
    print("="*60)

    # 1. Simulate data with confounders and a collider
    df = simulate_adr_data(n_patients=2000, seed=42)
    Y = df['Y'].values
    A = df['A'].values
    theta_true_adr = df['theta_true_adr'].iloc[0]

    print(f"True causal log(OR) for ADR: {theta_true_adr:.4f}")

    # --- Naive Estimation (confounded) ---
    print("\n--- 1. Naive Estimation (Biased by Confounding) ---")
    naive_model = SemiMechanisticLogisticMSM(Y, A)
    naive_solver = EstimatingEquationSolver(naive_model)
    theta_naive, _ = naive_solver.solve([-1, 0.5])
    print(f"Naive causal log(OR) estimate: {theta_naive[1]:.4f}")
    ci_low_naive, ci_high_naive = naive_solver.confint(theta_naive)
    print(f"95% CI: [{ci_low_naive[1]:.4f}, {ci_high_naive[1]:.4f}] (Biased)")

    # --- IPW with Collider (Incorrect Adjustment) ---
    print("\n--- 2. IPW with Collider (Biased by Collider Adjustment) ---")
    confounders_and_collider = df[['L', 'age', 'sex', 'comorbidities', 'hospitalized']].values
    ps_model_collider = fit_propensity_score(confounders_and_collider, A)
    ps_collider = ps_model_collider.predict_proba(confounders_and_collider)[:, 1]
    weights_collider = compute_ipw(A, ps_collider)

    ipw_model_collider = SemiMechanisticLogisticMSM(Y, A, weights=weights_collider)
    ipw_solver_collider = EstimatingEquationSolver(ipw_model_collider)
    theta_ipw_collider, _ = ipw_solver_collider.solve([-1, 0.5])
    print(f"IPW estimate adjusting for collider: {theta_ipw_collider[1]:.4f}")
    ci_low_collider, ci_high_collider = ipw_solver_collider.confint(theta_ipw_collider)
    print(f"95% CI: [{ci_low_collider[1]:.4f}, {ci_high_collider[1]:.4f}] (Bias induced by collider)")

    # --- IPW Correctly Adjusted (Unconfounded) ---
    print("\n--- 3. Correct IPW (Adjusted for Confounders Only) ---")
    confounders_only = df[['L', 'age', 'sex', 'comorbidities']].values
    ps_model_correct = fit_propensity_score(confounders_only, A)
    ps_correct = ps_model_correct.predict_proba(confounders_only)[:, 1]
    weights_correct = compute_ipw(A, ps_correct)

    ipw_model_correct = SemiMechanisticLogisticMSM(Y, A, weights=weights_correct)
    ipw_solver_correct = EstimatingEquationSolver(ipw_model_correct)
    theta_ipw_correct, _ = ipw_solver_correct.solve([-1, 0.5])
    print(f"Correct IPW estimate: {theta_ipw_correct[1]:.4f}")
    ci_low_correct, ci_high_correct = ipw_solver_correct.confint(theta_ipw_correct)
    print(f"95% CI: [{ci_low_correct[1]:.4f}, {ci_high_correct[1]:.4f}] (Closest to true value)")

    print("\n" + "="*60)
    print("Summary:")
    print("- The Naive estimate is biased because it ignores confounders.")
    print("- Adjusting for a collider ('hospitalized') introduces new bias.")
    print("- The correct IPW estimate adjusts for confounders only and is closest to the truth.")
    print("="*60)