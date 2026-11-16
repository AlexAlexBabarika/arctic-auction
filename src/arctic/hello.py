"""Hello-world check: tiny Fisher market (apple, bread; Alice, Bob)."""

import sys

import numpy as np

BUDGETS = np.array([2.0, 1.0])
VALUES = np.array([[3.0, 1.0], [1.0, 2.0]])  # rows: Alice, Bob; cols: apple, bread
SUPPLY = np.array([1.0, 1.0])
EXPECTED = np.array([2.0, 1.0])
TOL = 1e-3


def close(prices) -> bool:
    return bool(np.allclose(prices, EXPECTED, atol=TOL))


def eisenberg_gale_prices(solver: str) -> np.ndarray:
    """Solve the Eisenberg-Gale program with CVXPY; prices = supply duals."""
    import cvxpy as cp

    x = cp.Variable(VALUES.shape, nonneg=True)
    utility = cp.sum(cp.multiply(VALUES, x), axis=1)
    supply = [cp.sum(x[:, j]) <= SUPPLY[j] for j in range(len(SUPPLY))]
    problem = cp.Problem(cp.Maximize(BUDGETS @ cp.log(utility)), supply)
    problem.solve(solver=solver)
    return np.array([float(c.dual_value) for c in supply])


def proportional_response_prices(rounds: int = 200) -> np.ndarray:
    """Proportional response dynamics in JAX (float64, jitted update)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    budgets = jnp.array(BUDGETS)
    values = jnp.array(VALUES)

    @jax.jit
    def step(bids):
        prices = bids.sum(axis=0)
        alloc = bids / prices
        utility = (values * alloc).sum(axis=1, keepdims=True)
        return budgets[:, None] * values * alloc / utility

    bids = jnp.tile(budgets[:, None] / values.shape[1], (1, values.shape[1]))
    for _ in range(rounds):
        bids = step(bids)
    return np.asarray(bids.sum(axis=0))


def check_cvxpy(solver: str):
    try:
        prices = eisenberg_gale_prices(solver)
    except Exception as e:  # never let one solver crash the script
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        if solver == "MOSEK" and ("licen" in str(e).lower() or "mosek" in str(e).lower()):
            return "SKIP", f"MOSEK unavailable (likely missing licence): {msg}"
        return "FAIL", f"{type(e).__name__}: {msg}"
    return ("PASS" if close(prices) else "FAIL"), f"prices = {np.round(prices, 4)}"


def check_gurobi():
    try:
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as m:
                x = m.addVar(lb=0)
                y = m.addVar(lb=0)
                m.setObjective(x + y, gp.GRB.MAXIMIZE)
                m.addConstr(x + y <= 3)
                m.optimize()
                val = m.ObjVal
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}"
    return ("PASS" if abs(val - 3) < 1e-6 else "FAIL"), f"objective = {val}"


def check_highs():
    try:
        import highspy

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        inf = highspy.kHighsInf
        h.addVar(0, inf)
        h.addVar(0, inf)
        h.changeColCost(0, 1)
        h.changeColCost(1, 1)
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        h.addRow(-inf, 3, 2, np.array([0, 1], dtype=np.int32), np.array([1.0, 1.0]))
        h.run()
        val = h.getInfo().objective_function_value
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}"
    return ("PASS" if abs(val - 3) < 1e-6 else "FAIL"), f"objective = {val}"


def check_jax():
    try:
        import jax

        prices = proportional_response_prices()
        device = jax.devices()[0].platform
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}"
    return ("PASS" if close(prices) else "FAIL"), f"prices = {np.round(prices, 4)}, device = {device}"


def main() -> int:
    print(f"Expected prices: apple {EXPECTED[0]:.2f}, bread {EXPECTED[1]:.2f}\n")
    checks = [
        ("CVXPY + MOSEK", lambda: check_cvxpy("MOSEK")),
        ("CVXPY + Clarabel", lambda: check_cvxpy("CLARABEL")),
        ("CVXPY + SCS", lambda: check_cvxpy("SCS")),
        ("Gurobi", check_gurobi),
        ("HiGHS", check_highs),
        ("JAX", check_jax),
    ]
    results = []
    for name, fn in checks:
        status, detail = fn()
        print(f"[{status}] {name}: {detail}")
        results.append((name, status))

    print("\nSummary")
    print("-" * 28)
    for name, status in results:
        print(f"{name:<18}{status}")
    return 1 if any(s == "FAIL" for _, s in results) else 0


if __name__ == "__main__":
    sys.exit(main())
