import numpy as np

from SimulatorEngine import DupireSimulator


def synthetic_iv_surface(tenors, strikes, s0):
    t_grid, k_grid = np.meshgrid(tenors, strikes, indexing="ij")
    m = np.log(np.maximum(k_grid, 1e-12) / max(s0, 1e-12))
    iv = 0.16 + 0.05 * np.sqrt(np.maximum(t_grid, 1e-12)) - 0.04 * m + 0.08 * (m ** 2)
    return np.clip(iv, 0.05, 0.9)


def main():
    s0 = 100.0
    r = 0.02
    q = 0.00
    tenors = np.array([0.25, 0.5, 1.0, 1.5, 2.0], dtype=float)
    strikes = np.array([75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0], dtype=float)
    iv_mkt = synthetic_iv_surface(tenors, strikes, s0)

    sim = DupireSimulator(S0=s0, r_zero=r, q_zero=q, n_assets=1)
    _, diag = sim.calibrate_v2(
        vol_matrices=iv_mkt,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
        degree=3,
        n_knots_T=3,
        n_knots_K=4,
        alpha=1e-2,
        lm_lambda0=1e-1,
        max_iter=5,
        tol=1e-5,
        use_log_sigma=True,
        nK_pde=140,
        nT_pde=200,
        pde_theta=1.0,
        return_diagnostics=True,
    )

    d = diag["per_asset"][0]
    iv_model = np.asarray(d["iv_model_final"], dtype=float)
    iv_err = iv_model - iv_mkt
    mae = float(np.nanmean(np.abs(iv_err)))
    rmse = float(np.sqrt(np.nanmean(iv_err * iv_err)))

    print("=== calibrate_v2 (IV residual) minimal demo ===")
    print(f"Objective initial: {d['objective_initial']:.6e}")
    print(f"Objective final:   {d['objective_final']:.6e}")
    print(f"IV MAE final:      {mae:.6e}")
    print(f"IV RMSE final:     {rmse:.6e}")
    print(f"IV inversion fails initial/final: {d['n_fail_iv_inversions_initial']} -> {d['n_fail_iv_inversions_final']}")
    print("Convergence history (iter, obj, lambda, rho, accepted):")
    for h in d["lm_history"]:
        print(
            f"  {h['iter']:02d}  obj={h['obj']:.6e}  lam={h['lambda']:.3e}  "
            f"rho={h['rho']:.3e}  acc={h['accepted']}"
        )


if __name__ == "__main__":
    main()

