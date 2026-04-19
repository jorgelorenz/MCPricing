import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from SimulatorEngine import DupireSimulator
from unrisk_adapter import (
    list_equity_assets,
    read_implied_for_engine,
    read_local_for_engine,
    write_engine_local_to_unrisk_json,
)


def bs_call_price(s0, k, r, q, sigma, t):
    t_eff = max(float(t), 1e-12)
    sig = max(float(sigma), 1e-12)
    k_eff = max(float(k), 1e-12)
    sqrt_t = np.sqrt(t_eff)
    d1 = (np.log(max(s0, 1e-12) / k_eff) + (r - q + 0.5 * sig * sig) * t_eff) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    return s0 * np.exp(-q * t_eff) * ndtr(d1) - k_eff * np.exp(-r * t_eff) * ndtr(d2)


def bs_implied_vol_from_price(s0, k, r, q, t, call_price, vol_low=1e-6, vol_high=5.0):
    t_eff = max(float(t), 1e-12)
    k_eff = max(float(k), 1e-12)
    df_r = np.exp(-r * t_eff)
    df_q = np.exp(-q * t_eff)
    intrinsic = max(s0 * df_q - k_eff * df_r, 0.0)
    upper = s0 * df_q
    target = float(np.clip(call_price, intrinsic + 1e-14, upper - 1e-14))
    extrinsic = target - intrinsic

    # Degenerate regime for IV inversion: tiny time value or ultra-short maturity.
    extrinsic_abs_tol = 1e-4
    extrinsic_rel_tol = 1e-6
    t_min_for_iv = 7.0 / 365.0
    if t_eff < t_min_for_iv or extrinsic <= max(extrinsic_abs_tol, extrinsic_rel_tol * max(upper, 1.0)):
        return np.nan

    def f(sig):
        return bs_call_price(s0=s0, k=k_eff, r=r, q=q, sigma=sig, t=t_eff) - target

    f_low = f(vol_low)
    f_high = f(vol_high)
    if f_low * f_high > 0.0:
        return np.nan
    return float(brentq(f, vol_low, vol_high, xtol=1e-12, maxiter=200))


def price_grid_from_implied(s0, tenors, strikes, implied_matrix, r_curve, q_curve, integrator_fn):
    int_r = integrator_fn(r_curve, np.asarray(tenors, dtype=float))
    int_q = integrator_fn(q_curve, np.asarray(tenors, dtype=float))
    prices = np.empty_like(implied_matrix, dtype=float)
    for i, t in enumerate(tenors):
        r_eff = float(int_r[i] / max(t, 1e-12))
        q_eff = float(int_q[i] / max(t, 1e-12))
        for j, k in enumerate(strikes):
            prices[i, j] = bs_call_price(s0=s0, k=k, r=r_eff, q=q_eff, sigma=float(implied_matrix[i, j]), t=t)
    return prices


def implied_grid_from_prices(s0, tenors, strikes, prices, r_curve, q_curve, integrator_fn):
    int_r = integrator_fn(r_curve, np.asarray(tenors, dtype=float))
    int_q = integrator_fn(q_curve, np.asarray(tenors, dtype=float))
    out = np.empty_like(prices, dtype=float)
    reliable = np.zeros_like(prices, dtype=bool)
    extrinsic = np.empty_like(prices, dtype=float)
    for i, t in enumerate(tenors):
        r_eff = float(int_r[i] / max(t, 1e-12))
        q_eff = float(int_q[i] / max(t, 1e-12))
        df_r = np.exp(-r_eff * float(t))
        df_q = np.exp(-q_eff * float(t))
        for j, k in enumerate(strikes):
            intrinsic = max(float(s0) * df_q - float(k) * df_r, 0.0)
            extrinsic[i, j] = float(prices[i, j]) - intrinsic
            iv = bs_implied_vol_from_price(
                s0=s0, k=k, r=r_eff, q=q_eff, t=t, call_price=float(prices[i, j])
            )
            out[i, j] = iv
            reliable[i, j] = bool(np.isfinite(iv))
    return out, reliable, extrinsic


def plot_iv_compare(strikes, tenors, iv_market, iv_model, iv_local_read, out_path):
    n_t = len(tenors)
    n_cols = 3
    n_rows = int(np.ceil(n_t / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)
    for i, t in enumerate(tenors):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(strikes, iv_market[i, :], "o-", label="IV market", linewidth=1.8, markersize=4)
        ax.plot(strikes, iv_model[i, :], "s--", label="IV model", linewidth=1.5, markersize=4)
        ax.plot(strikes, iv_local_read[i, :], "^-.", label="IV local-read", linewidth=1.3, markersize=4)
        ax.set_title(f"T={t:.3f}y")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Vol")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    for j in range(n_t, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calibra 1 subyacente desde JSON UnRisk y compara IV market vs model")
    parser.add_argument("--path", default="datos/EquityBasket_8e55875c81554f3697bd765e672c3704_20260305_170633.json")
    parser.add_argument("--asset-index", type=int, default=0, help="Indice del subyacente a usar (0-based)")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Ruta de salida JSON con la local calibrada inyectada (default: <input>_with_engine_local.json)",
    )
    args = parser.parse_args()

    catalog = list_equity_assets(args.path)
    if args.asset_index < 0 or args.asset_index >= len(catalog):
        raise IndexError(f"--asset-index {args.asset_index} fuera de rango [0, {len(catalog)-1}]")
    print("=== Catalogo de subyacentes ===")
    for a in catalog:
        marker = "*" if a["index"] == args.asset_index else " "
        print(f"{marker} {a['index']:2d} | {a['name']} | ccy={a['currency']}")
    print("")

    raw = read_implied_for_engine(args.path, asset_index=args.asset_index)
    raw_local = read_local_for_engine(args.path, asset_index=args.asset_index)
    asset_name = raw["asset_names"][0]

    s0 = float(raw["S0_list"][0])
    tenors = np.asarray(raw["tenors_list"][0], dtype=float)
    strikes = np.asarray(raw["strikes_list"][0], dtype=float)
    iv_market = np.asarray(raw["vol_matrices"][0], dtype=float)
    r_curve = raw["r_zero_list"][0]
    q_curve = raw["q_zero_list"][0]
    local_read_matrix = np.asarray(raw_local["vol_matrices"][0], dtype=float)

    sim = DupireSimulator(S0=s0, r_zero=r_curve, q_zero=q_curve, n_assets=1)
    t0 = perf_counter()
    sim.calibrate_v2(
        vol_matrices=iv_market,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
        degree=3,
        n_knots_T=4,
        n_knots_K=8,
        alpha=1e-2,
        lm_lambda0=1e-1,
        max_iter=8,
        tol=1e-6,
        use_log_sigma=True,
        nK_pde=400,
        nT_pde=500,
        pde_theta=1.0,
    )
    t1 = perf_counter()

    sigma_fn = sim.vol_interpolators_[0]
    model_prices = sim._dupire_solve_call_surface(
        S0=s0,
        tenors=tenors,
        strikes=strikes,
        sigma_fn=sigma_fn,
        r_curve=r_curve,
        q_curve=q_curve,
        vol_floor=1e-8,
        nK_pde=600,
        nT_pde=800,
        theta=1.0,
    )

    market_prices = price_grid_from_implied(
        s0=s0,
        tenors=tenors,
        strikes=strikes,
        implied_matrix=iv_market,
        r_curve=r_curve,
        q_curve=q_curve,
        integrator_fn=sim._integrate_curve,
    )

    iv_model, rel_model, extr_model = implied_grid_from_prices(
        s0=s0,
        tenors=tenors,
        strikes=strikes,
        prices=model_prices,
        r_curve=r_curve,
        q_curve=q_curve,
        integrator_fn=sim._integrate_curve,
    )

    sim_local_read = DupireSimulator(
        S0=s0,
        r_zero=r_curve,
        q_zero=q_curve,
        vol_matrices=local_read_matrix,
        tenor_grids=tenors,
        strike_grids=strikes,
        n_assets=1,
    )
    local_read_prices = sim_local_read._dupire_solve_call_surface(
        S0=s0,
        tenors=tenors,
        strikes=strikes,
        sigma_fn=sim_local_read.vol_interpolators_[0],
        r_curve=r_curve,
        q_curve=q_curve,
        vol_floor=1e-8,
        nK_pde=600,
        nT_pde=800,
        theta=1.0,
    )
    iv_local_read, rel_local, extr_local = implied_grid_from_prices(
        s0=s0,
        tenors=tenors,
        strikes=strikes,
        prices=local_read_prices,
        r_curve=r_curve,
        q_curve=q_curve,
        integrator_fn=sim._integrate_curve,
    )

    price_err = model_prices - market_prices
    price_err_local_read = local_read_prices - market_prices
    iv_err = iv_model - iv_market
    iv_err_local_read = iv_local_read - iv_market
    mae_price = float(np.mean(np.abs(price_err)))
    rmse_price = float(np.sqrt(np.mean(price_err * price_err)))
    mae_price_local_read = float(np.mean(np.abs(price_err_local_read)))
    rmse_price_local_read = float(np.sqrt(np.mean(price_err_local_read * price_err_local_read)))
    mae_iv = float(np.nanmean(np.abs(iv_err)))
    rmse_iv = float(np.sqrt(np.nanmean(iv_err * iv_err)))
    mae_iv_local_read = float(np.nanmean(np.abs(iv_err_local_read)))
    rmse_iv_local_read = float(np.sqrt(np.nanmean(iv_err_local_read * iv_err_local_read)))
    rel_ratio_model = 100.0 * float(np.mean(rel_model))
    rel_ratio_local = 100.0 * float(np.mean(rel_local))
    iv_penalty = 0.5

    abs_err_model_strict = np.where(rel_model, np.abs(iv_err), iv_penalty)
    abs_err_local_strict = np.where(rel_local, np.abs(iv_err_local_read), iv_penalty)
    sq_err_model_strict = np.where(rel_model, iv_err * iv_err, iv_penalty * iv_penalty)
    sq_err_local_strict = np.where(rel_local, iv_err_local_read * iv_err_local_read, iv_penalty * iv_penalty)
    mae_iv_strict = float(np.mean(abs_err_model_strict))
    rmse_iv_strict = float(np.sqrt(np.mean(sq_err_model_strict)))
    mae_iv_local_strict = float(np.mean(abs_err_local_strict))
    rmse_iv_local_strict = float(np.sqrt(np.mean(sq_err_local_strict)))

    out_png = Path(f"unrisk_iv_compare_{asset_name.replace(' ', '_')}.png")
    plot_iv_compare(strikes, tenors, iv_market, iv_model, iv_local_read, out_png)

    print("=== UnRisk single-asset calibration demo ===")
    print(f"File: {args.path}")
    print(f"Asset: {asset_name}")
    print(f"Grid: tenors={tenors.size}, strikes={strikes.size}, shape={iv_market.shape}")
    print(f"Calibration time (v2): {t1 - t0:.3f} s")
    print(f"Price MAE (model vs bs-from-market-iv): {mae_price:.6e}")
    print(f"Price RMSE(model vs bs-from-market-iv): {rmse_price:.6e}")
    print(f"Price MAE (local-read vs bs-from-market-iv): {mae_price_local_read:.6e}")
    print(f"Price RMSE(local-read vs bs-from-market-iv): {rmse_price_local_read:.6e}")
    print(f"IV MAE (model vs market): {mae_iv:.6e}")
    print(f"IV RMSE(model vs market): {rmse_iv:.6e}")
    print(f"IV MAE (local-read vs market): {mae_iv_local_read:.6e}")
    print(f"IV RMSE(local-read vs market): {rmse_iv_local_read:.6e}")
    print(f"IV MAE strict (model, pen={iv_penalty}): {mae_iv_strict:.6e}")
    print(f"IV RMSE strict(model, pen={iv_penalty}): {rmse_iv_strict:.6e}")
    print(f"IV MAE strict (local, pen={iv_penalty}): {mae_iv_local_strict:.6e}")
    print(f"IV RMSE strict(local, pen={iv_penalty}): {rmse_iv_local_strict:.6e}")
    print(f"IV fiables model: {rel_ratio_model:.2f}%")
    print(f"IV fiables local-read: {rel_ratio_local:.2f}%")
    print("")
    print("T        K      IV_mkt    IV_model   iv_fiable   extr_model    IV_local   iv_fiable   extr_local")
    for i, t in enumerate(tenors):
        for j, k in enumerate(strikes):
            ivm = float(iv_market[i, j])
            ivmod = iv_model[i, j]
            ivloc = iv_local_read[i, j]
            ivmod_s = f"{ivmod:.6f}" if np.isfinite(ivmod) else "nan"
            ivloc_s = f"{ivloc:.6f}" if np.isfinite(ivloc) else "nan"
            print(
                f"{t:5.3f}  {k:8.3f}  {ivm:8.6f}  {ivmod_s:>8}  {str(bool(rel_model[i, j])):>9}  "
                f"{extr_model[i, j]:11.6e}  {ivloc_s:>8}  {str(bool(rel_local[i, j])):>9}  {extr_local[i, j]:11.6e}"
            )
    local_engine_obj = sim.local_vol_matrices_[0]
    if hasattr(local_engine_obj, "values"):
        local_engine_matrix = np.asarray(local_engine_obj.values, dtype=float)
    else:
        local_engine_matrix = np.asarray(local_engine_obj, dtype=float)
    exported_json = write_engine_local_to_unrisk_json(
        path=args.path,
        tenors=tenors,
        strikes=strikes,
        local_vol_matrix=local_engine_matrix,
        asset_index=args.asset_index,
        out_path=args.out_json,
    )
    print(f"Plot: {out_png}")
    print(f"Exported local JSON: {exported_json}")


if __name__ == "__main__":
    main()


