import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from time import perf_counter
import matplotlib.pyplot as plt

from SimulatorEngine import DupireSimulator
from Utils import LocalVolatilityMatrix


def bs_call_price(s0, k, r, q, sigma, t):
    t_eff = max(float(t), 1e-12)
    sig = max(float(sigma), 1e-12)
    sqrt_t = np.sqrt(t_eff)
    d1 = (np.log(max(s0, 1e-12) / max(k, 1e-12)) + (r - q + 0.5 * sig * sig) * t_eff) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    return s0 * np.exp(-q * t_eff) * ndtr(d1) - k * np.exp(-r * t_eff) * ndtr(d2)


def bs_implied_vol_from_price(s0, k, r, q, t, call_price, vol_low=1e-6, vol_high=5.0):
    t_eff = max(float(t), 1e-12)
    k_eff = max(float(k), 1e-12)
    disc_r = np.exp(-r * t_eff)
    disc_q = np.exp(-q * t_eff)
    intrinsic = max(s0 * disc_q - k_eff * disc_r, 0.0)
    upper = s0 * disc_q
    target = float(np.clip(call_price, intrinsic + 1e-14, upper - 1e-14))

    def f(sig):
        return bs_call_price(s0=s0, k=k_eff, r=r, q=q, sigma=sig, t=t_eff) - target

    f_low = f(vol_low)
    f_high = f(vol_high)
    if f_low * f_high > 0.0:
        return np.nan
    return float(brentq(f, vol_low, vol_high, maxiter=200, xtol=1e-12))


def implied_vol_matrix_from_prices(s0, r, q, tenors, strikes, call_prices):
    tenors = np.asarray(tenors, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    out = np.empty_like(call_prices, dtype=float)
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            out[i, j] = bs_implied_vol_from_price(s0=s0, k=k, r=r, q=q, t=t, call_price=call_prices[i, j])
    return out


def integral_curve(curve, maturity, n_steps=400):
    t = float(maturity)
    if t <= 0.0:
        return 0.0
    grid = np.linspace(0.0, t, int(n_steps) + 1)
    vals = np.asarray(curve(grid), dtype=float)
    if vals.shape == ():
        vals = np.full_like(grid, float(vals), dtype=float)
    return float(cumulative_trapezoid(vals, grid, initial=0.0)[-1])


def plot_implied_by_tenor(strikes, tenors, iv_mkt, iv_v1, iv_v2, out_path="iv_by_tenor.png"):
    n_t = len(tenors)
    n_cols = 3
    n_rows = int(np.ceil(n_t / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)

    for i, t in enumerate(tenors):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(strikes, iv_mkt[i, :], "o-", label="IV mkt", linewidth=1.8, markersize=4)
        ax.plot(strikes, iv_v1[i, :], "s--", label="IV v1", linewidth=1.5, markersize=4)
        ax.plot(strikes, iv_v2[i, :], "d-.", label="IV v2", linewidth=1.5, markersize=4)
        ax.set_title(f"T={t:.2f}y")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Vol")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for j in range(n_t, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_vol_surfaces(tenors, strikes, s0):
    t_grid, k_grid = np.meshgrid(tenors, strikes, indexing="ij")
    moneyness = np.log(np.maximum(k_grid, 1e-12) / max(s0, 1e-12))

    # Superficie implícita sintética: nivel + term-structure + skew + smile
    implied_vol = (
        0.16
        + 0.04 * np.sqrt(np.maximum(t_grid, 1e-12))
        - 0.05 * moneyness
        + 0.10 * (moneyness ** 2)
    )
    implied_vol = np.clip(implied_vol, 0.05, 0.80)

    # Superficie local base de referencia (no usada para calibrar, solo para inspección).
    local_vol_ref = np.clip(implied_vol * (1.0 + 0.03 * np.sin(2.0 * np.pi * t_grid)), 0.05, 0.90)

    return implied_vol, local_vol_ref


def price_call_dupire_mc(simulator, k, t, r, n_sims=50000, seed=12345):
    np.random.seed(seed)
    paths = simulator.simulate([t], n_sims=n_sims)
    st = paths[:, -1]
    payoff = np.maximum(st - k, 0.0)
    return np.exp(-r * t) * np.mean(payoff)


def price_calls_dupire_mc_grid(simulator, strikes, maturities, r, n_sims=300000, seed=12345):
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    order = np.argsort(maturities)
    mats_sorted = maturities[order]

    np.random.seed(seed)
    paths = simulator.simulate(list(mats_sorted), n_sims=n_sims)
    prices_sorted = np.empty((mats_sorted.size, strikes.size), dtype=float)
    for it, t in enumerate(mats_sorted):
        st = paths[:, it]
        payoff = np.maximum(st[:, None] - strikes[None, :], 0.0)
        prices_sorted[it, :] = np.exp(-r * t) * np.mean(payoff, axis=0)

    inv = np.argsort(order)
    return prices_sorted[inv, :]


def price_calls_dupire_pde_grid(simulator, strikes, maturities, r, q, nK_pde=260, nT_pde=360, pde_theta=1.0):
    sigma_fn = simulator.vol_interpolators_[0]
    return simulator._dupire_solve_call_surface(
        S0=float(simulator.S0_ if np.isscalar(simulator.S0_) else simulator.S0_[0]),
        tenors=np.asarray(maturities, dtype=float),
        strikes=np.asarray(strikes, dtype=float),
        sigma_fn=sigma_fn,
        r_curve=r,
        q_curve=q,
        vol_floor=1e-8,
        nK_pde=nK_pde,
        nT_pde=nT_pde,
        theta=float(pde_theta),
    )


def price_constant_local_vol_pde(simulator, sigma_const, strike, maturity, r, q, nK_pde=400, nT_pde=500, pde_theta=1.0):
    sigma_const = float(sigma_const)

    def sigma_fn(points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be shape (n,2)")
        return np.full(pts.shape[0], sigma_const, dtype=float)

    out = simulator._dupire_solve_call_surface(
        S0=float(simulator.S0_ if np.isscalar(simulator.S0_) else simulator.S0_[0]),
        tenors=np.array([float(maturity)], dtype=float),
        strikes=np.array([float(strike)], dtype=float),
        sigma_fn=sigma_fn,
        r_curve=r,
        q_curve=q,
        vol_floor=1e-12,
        nK_pde=nK_pde,
        nT_pde=nT_pde,
        theta=float(pde_theta),
    )
    return float(out[0, 0])


def price_constant_local_vol_pde_grid(simulator, sigma_const, strikes, maturities, r_curve, q_curve, nK_pde=380, nT_pde=520, pde_theta=1.0):
    sigma_const = float(sigma_const)

    def sigma_fn(points):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be shape (n,2)")
        return np.full(pts.shape[0], sigma_const, dtype=float)

    out = simulator._dupire_solve_call_surface(
        S0=float(simulator.S0_ if np.isscalar(simulator.S0_) else simulator.S0_[0]),
        tenors=np.asarray(maturities, dtype=float),
        strikes=np.asarray(strikes, dtype=float),
        sigma_fn=sigma_fn,
        r_curve=r_curve,
        q_curve=q_curve,
        vol_floor=1e-12,
        nK_pde=nK_pde,
        nT_pde=nT_pde,
        theta=float(pde_theta),
    )
    return out


def main():
    s0 = 100.0
    r = 0.02
    q = 0.00

    tenors = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00], dtype=float)
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0], dtype=float)

    implied_vol, local_vol_ref = generate_vol_surfaces(tenors, strikes, s0)

    dupire = DupireSimulator(S0=s0, r_zero=r, q_zero=q, n_assets=1)
    t0 = perf_counter()
    dupire.calibrate(
        vol_matrices=implied_vol,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
        max_iter=100000,
        smooth_w=0.0005,
        lr=0.05,
        tol=0.0
    )
    t1 = perf_counter()

    dupire_v2 = DupireSimulator(S0=s0, r_zero=r, q_zero=q, n_assets=1)
    t2 = perf_counter()
    _, diag_v2 = dupire_v2.calibrate_v2(
        vol_matrices=implied_vol,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
        degree=3,
        n_knots_T=3,
        n_knots_K=5,
        alpha=1e-2,
        lm_lambda0=1e-1,
        max_iter=10,
        tol=1e-5,
        use_log_sigma=True,
        nK_pde=160,
        nT_pde=220,
        pde_theta=1.0,
        return_diagnostics=True
    )
    t3 = perf_counter()

    # Rejilla de test dentro de la malla de calibración.
    test_tenors = np.array([0.50, 1.00, 1.50, 2.00], dtype=float)
    test_strikes = np.array([75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0], dtype=float)

    # Precios BS de referencia usando la superficie implied interpolada.
    iv_interp = LocalVolatilityMatrix(tenors, strikes, implied_vol, vol_floor=1e-8)
    price_bs_grid = np.empty((test_tenors.size, test_strikes.size), dtype=float)
    sigma_bs_grid = np.empty_like(price_bs_grid)
    for i, t in enumerate(test_tenors):
        for j, k in enumerate(test_strikes):
            sigma_ref = float(iv_interp(t, k))
            sigma_bs_grid[i, j] = sigma_ref
            price_bs_grid[i, j] = bs_call_price(s0=s0, k=k, r=r, q=q, sigma=sigma_ref, t=t)

    pricing_mode = "pde"  # "pde" | "mc"
    if pricing_mode == "mc":
        n_sims = 500000
        price_dupire_grid = price_calls_dupire_mc_grid(
            dupire, strikes=test_strikes, maturities=test_tenors, r=r, n_sims=n_sims, seed=12345
        )
        price_dupire_v2_grid = price_calls_dupire_mc_grid(
            dupire_v2, strikes=test_strikes, maturities=test_tenors, r=r, n_sims=n_sims, seed=12345
        )
        pricing_label = f"MC (n_sims={n_sims})"
    else:
        pde_price_nK = 260
        pde_price_nT = 360
        price_dupire_grid = price_calls_dupire_pde_grid(
            dupire, strikes=test_strikes, maturities=test_tenors, r=r, q=q, nK_pde=pde_price_nK, nT_pde=pde_price_nT, pde_theta=1.0
        )
        price_dupire_v2_grid = price_calls_dupire_pde_grid(
            dupire_v2, strikes=test_strikes, maturities=test_tenors, r=r, q=q, nK_pde=pde_price_nK, nT_pde=pde_price_nT, pde_theta=1.0
        )
        pricing_label = f"PDE (nK={pde_price_nK}, nT={pde_price_nT}, theta=1.0)"

    d0 = diag_v2["per_asset"][0]
    obj_ini = float(d0["objective_initial"])
    obj_fin = float(d0["objective_final"])
    obj_impr = obj_ini - obj_fin
    lv_min = float(np.min(dupire_v2.local_vol_matrices_[0].values))
    lv_pos = bool(lv_min > 0.0)

    err_v1 = price_dupire_grid - price_bs_grid
    err_v2 = price_dupire_v2_grid - price_bs_grid

    mae_v1 = float(np.mean(np.abs(err_v1)))
    mae_v2 = float(np.mean(np.abs(err_v2)))
    rmse_v1 = float(np.sqrt(np.mean(err_v1 * err_v1)))
    rmse_v2 = float(np.sqrt(np.mean(err_v2 * err_v2)))
    maxae_v1 = float(np.max(np.abs(err_v1)))
    maxae_v2 = float(np.max(np.abs(err_v2)))

    rel_v1 = np.abs(err_v1) / np.maximum(np.abs(price_bs_grid), 1e-12)
    rel_v2 = np.abs(err_v2) / np.maximum(np.abs(price_bs_grid), 1e-12)
    mape_v1 = float(100.0 * np.mean(rel_v1))
    mape_v2 = float(100.0 * np.mean(rel_v2))

    print("=== Superficies generadas ===")
    print(f"Implied vol matrix shape: {implied_vol.shape}")
    print(f"Local vol ref matrix shape: {local_vol_ref.shape}")
    print("")
    print("=== Prueba en múltiples strikes y vencimientos ===")
    print(f"S0={s0:.4f}, r={r:.4f}, q={q:.4f}, pricing={pricing_label}")
    print(f"Tiempo calibracion v1: {(t1 - t0):.3f} s")
    print(f"Tiempo calibracion v2: {(t3 - t2):.3f} s")
    print(f"Test tenors:  {test_tenors}")
    print(f"Test strikes: {test_strikes}")
    print(f"MAE   v1 vs BS: {mae_v1:.6f}")
    print(f"MAE   v2 vs BS: {mae_v2:.6f}")
    print(f"RMSE  v1 vs BS: {rmse_v1:.6f}")
    print(f"RMSE  v2 vs BS: {rmse_v2:.6f}")
    print(f"MaxAE v1 vs BS: {maxae_v1:.6f}")
    print(f"MaxAE v2 vs BS: {maxae_v2:.6f}")
    print(f"MAPE  v1 vs BS: {mape_v1:.3f}%")
    print(f"MAPE  v2 vs BS: {mape_v2:.3f}%")
    print("")
    print("=== Detalle por punto (T,K): BS vs v1 vs v2 ===")
    print("    T        K           BS           v1           v2     |v1-BS|     |v2-BS|")
    for i, t in enumerate(test_tenors):
        for j, k in enumerate(test_strikes):
            pbs = float(price_bs_grid[i, j])
            pv1 = float(price_dupire_grid[i, j])
            pv2 = float(price_dupire_v2_grid[i, j])
            e1 = abs(pv1 - pbs)
            e2 = abs(pv2 - pbs)
            print(f"{t:6.3f}  {k:7.2f}  {pbs:11.6f}  {pv1:11.6f}  {pv2:11.6f}  {e1:10.6f}  {e2:10.6f}")
    print("")
    print("=== Diagnostico calibrate_v2 ===")
    print(f"Objetivo inicial:           {obj_ini:.6e}")
    print(f"Objetivo final:             {obj_fin:.6e}")
    print(f"Mejora objetivo:            {obj_impr:.6e}")
    print(f"Sigma minima (malla mkt):   {lv_min:.6e}")
    print(f"Chequeo sigma > 0:          {lv_pos}")
    print("")
    print("=== Check PDE vs BS con local vol plana ===")
    t_chk = 1.00
    k_chk = 100.0
    sigma_chk = float(iv_interp(t_chk, k_chk))
    price_bs_chk = bs_call_price(s0=s0, k=k_chk, r=r, q=q, sigma=sigma_chk, t=t_chk)
    price_pde_chk = price_constant_local_vol_pde(
        dupire_v2, sigma_const=sigma_chk, strike=k_chk, maturity=t_chk, r=r, q=q,
        nK_pde=500, nT_pde=600, pde_theta=1.0
    )
    abs_chk = abs(price_pde_chk - price_bs_chk)
    rel_chk = abs_chk / max(abs(price_bs_chk), 1e-12)
    print(f"Nodo check: T={t_chk:.3f}, K={k_chk:.3f}, sigma={sigma_chk:.6f}")
    print(f"Precio BS (sigma plana):    {price_bs_chk:.6f}")
    print(f"Precio PDE (sigma plana):   {price_pde_chk:.6f}")
    print(f"Diferencia abs PDE-BS:      {abs_chk:.6e}")
    print(f"Diferencia rel PDE-BS:      {100.0 * rel_chk:.4f}%")

    print("")
    print("=== Check curva no constante: BS vs PDE Dupire (vol plana) ===")
    r_curve_ns = lambda t: 0.012 + 0.010 * np.exp(-0.8 * np.asarray(t, dtype=float)) + 0.004 * np.asarray(t, dtype=float)
    q_curve_ns = lambda t: 0.006 + 0.003 * (1.0 - np.exp(-0.7 * np.asarray(t, dtype=float)))
    test_t_curve = np.array([0.50, 1.00, 1.50, 2.00], dtype=float)
    test_k_curve = np.array([90.0, 100.0, 110.0], dtype=float)
    sigma_const_curve = 0.22

    pde_curve_prices = price_constant_local_vol_pde_grid(
        dupire_v2,
        sigma_const=sigma_const_curve,
        strikes=test_k_curve,
        maturities=test_t_curve,
        r_curve=r_curve_ns,
        q_curve=q_curve_ns,
        nK_pde=420,
        nT_pde=620,
        pde_theta=1.0
    )
    bs_curve_prices = np.empty_like(pde_curve_prices)
    print("    T        K       r_eff(T)    q_eff(T)      BS_price    PDE_price    |diff|    rel%")
    for i, t in enumerate(test_t_curve):
        int_r_t = integral_curve(r_curve_ns, t)
        int_q_t = integral_curve(q_curve_ns, t)
        r_eff_t = int_r_t / t
        q_eff_t = int_q_t / t
        for j, k in enumerate(test_k_curve):
            bs_p = bs_call_price(s0=s0, k=k, r=r_eff_t, q=q_eff_t, sigma=sigma_const_curve, t=t)
            pde_p = float(pde_curve_prices[i, j])
            bs_curve_prices[i, j] = bs_p
            abs_d = abs(pde_p - bs_p)
            rel_d = 100.0 * abs_d / max(abs(bs_p), 1e-12)
            print(f"{t:6.3f}  {k:7.2f}  {r_eff_t:10.6f}  {q_eff_t:10.6f}  {bs_p:11.6f}  {pde_p:11.6f}  {abs_d:8.6f}  {rel_d:6.3f}")
    mae_curve = float(np.mean(np.abs(pde_curve_prices - bs_curve_prices)))
    rmse_curve = float(np.sqrt(np.mean((pde_curve_prices - bs_curve_prices) ** 2)))
    print(f"MAE curva (BS vs PDE):  {mae_curve:.6e}")
    print(f"RMSE curva (BS vs PDE): {rmse_curve:.6e}")

    print("")
    print("=== Plot IV por tenor (mkt vs v1 vs v2) ===")
    market_grid_prices = np.empty((tenors.size, strikes.size), dtype=float)
    for ii, tt in enumerate(tenors):
        for jj, kk in enumerate(strikes):
            market_grid_prices[ii, jj] = bs_call_price(
                s0=s0, k=kk, r=r, q=q, sigma=float(implied_vol[ii, jj]), t=float(tt)
            )

    model_prices_v1_grid = price_calls_dupire_pde_grid(
        dupire, strikes=strikes, maturities=tenors, r=r, q=q, nK_pde=260, nT_pde=360, pde_theta=1.0
    )
    model_prices_v2_grid = price_calls_dupire_pde_grid(
        dupire_v2, strikes=strikes, maturities=tenors, r=r, q=q, nK_pde=260, nT_pde=360, pde_theta=1.0
    )
    iv_v1_grid = implied_vol_matrix_from_prices(s0=s0, r=r, q=q, tenors=tenors, strikes=strikes, call_prices=model_prices_v1_grid)
    iv_v2_grid = implied_vol_matrix_from_prices(s0=s0, r=r, q=q, tenors=tenors, strikes=strikes, call_prices=model_prices_v2_grid)
    plot_path = plot_implied_by_tenor(
        strikes=strikes, tenors=tenors, iv_mkt=implied_vol, iv_v1=iv_v1_grid, iv_v2=iv_v2_grid,
        out_path="iv_by_tenor_v1_vs_v2.png"
    )
    print(f"Grafica guardada en: {plot_path}")


if __name__ == "__main__":
    main()
