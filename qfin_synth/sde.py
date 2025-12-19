import numpy as np
import pandas as pd
from tqdm import tqdm

def simulate_paths(
    est,  # MuSigmaNonParam
    start_price: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    ewma_alpha: float,
    burnin: int,
    init_sigma: float,
    seed: int = 123,
):
    """
    Euler–Maruyama simulation on log-price increments:

      r_t = mu(Z_{t-1})*dt + sigma(Z_{t-1})*sqrt(dt)*eps_t
      P_t = P_{t-1} * exp(r_t)

    State:
      Z_{t-1} = (r_{t-1}, lvol_{t-1})
      lvol_t = 0.5*log(sigma2_t)
      sigma2_t = (1-alpha)*sigma2_{t-1} + alpha*r_t^2
    """
    if not (0.0 < ewma_alpha < 1.0):
        raise ValueError("ewma_alpha must be in (0,1)")
    rng = np.random.default_rng(seed)

    sigma2_0 = max(init_sigma**2, 1e-12)
    lvol0 = 0.5 * np.log(sigma2_0)
    total = n_steps + burnin

    out_all = []
    for p in tqdm(range(n_paths), desc="Simulating paths", leave=False):
        P = np.empty(total, dtype=float)
        r = np.empty(total, dtype=float)
        lvol = np.empty(total, dtype=float)

        P[0] = float(start_price)
        r[0] = 0.0
        lvol[0] = lvol0

        for t in range(1, total):
            state = np.array([[r[t-1], lvol[t-1]]], dtype=float)
            mu = float(est.predict_mu(state))
            sg = float(est.predict_sigma(state))
            eps = rng.standard_normal()
            r[t] = mu * dt + sg * np.sqrt(dt) * eps

            v_prev = np.exp(2.0 * lvol[t-1])
            v_t = (1.0 - ewma_alpha) * v_prev + ewma_alpha * (r[t] ** 2)
            v_t = max(v_t, 1e-12)
            lvol[t] = 0.5 * np.log(v_t)

            P[t] = P[t-1] * np.exp(r[t])

        out = pd.DataFrame({"path": p, "t": np.arange(n_steps, dtype=int), "price": P[burnin:]})
        out_all.append(out)

    return pd.concat(out_all, ignore_index=True)
