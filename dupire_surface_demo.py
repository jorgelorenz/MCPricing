import numpy as np
from scipy.special import ndtr

from SimulatorEngine import DupireSimulator
from Utils import LocalVolatilityMatrix


def bs_call_price(s0, k, r, q, sigma, t):
    t_eff = max(float(t), 1e-12)
    sig = max(float(sigma), 1e-12)
    sqrt_t = np.sqrt(t_eff)
    d1 = (np.log(max(s0, 1e-12) / max(k, 1e-12)) + (r - q + 0.5 * sig * sig) * t_eff) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    return s0 * np.exp(-q * t_eff) * ndtr(d1) - k * np.exp(-r * t_eff) * ndtr(d2)


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


def main():
    s0 = 100.0
    r = 0.02
    q = 0.00

    tenors = np.array([0.25, 0.50, 0.75, 1.00, 1.50, 2.00], dtype=float)
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0], dtype=float)

    implied_vol, local_vol_ref = generate_vol_surfaces(tenors, strikes, s0)

    dupire = DupireSimulator(S0=s0, r_zero=r, q_zero=q, n_assets=1)
    dupire.calibrate(
        vol_matrices=implied_vol,
        strikes_list=strikes,
        tenors_list=tenors,
        vol_floor=1e-8,
    )

    # Opción call de prueba dentro de la malla.
    t_test = 1.00
    k_test = 100.0

    # Vol implícita de referencia por interpolación bilineal sobre la superficie generada.
    iv_interp = LocalVolatilityMatrix(tenors, strikes, implied_vol, vol_floor=1e-8)
    sigma_bs_ref = float(iv_interp(t_test, k_test))

    price_bs = bs_call_price(s0=s0, k=k_test, r=r, q=q, sigma=sigma_bs_ref, t=t_test)
    price_dupire = price_call_dupire_mc(dupire, k=k_test, t=t_test, r=r, n_sims=50000, seed=12345)

    abs_diff = abs(price_dupire - price_bs)
    rel_diff = abs_diff / max(abs(price_bs), 1e-12)

    print("=== Superficies generadas ===")
    print(f"Implied vol matrix shape: {implied_vol.shape}")
    print(f"Local vol ref matrix shape: {local_vol_ref.shape}")
    print("")
    print("=== Prueba de valoración call ===")
    print(f"S0={s0:.4f}, K={k_test:.4f}, T={t_test:.4f}, r={r:.4f}, q={q:.4f}")
    print(f"Sigma implícita ref(K,T): {sigma_bs_ref:.6f}")
    print(f"Precio Call BS(ref sigma): {price_bs:.6f}")
    print(f"Precio Call Dupire MC:      {price_dupire:.6f}")
    print(f"Diferencia abs:             {abs_diff:.6f}")
    print(f"Diferencia rel:             {100.0 * rel_diff:.3f}%")


if __name__ == "__main__":
    main()
