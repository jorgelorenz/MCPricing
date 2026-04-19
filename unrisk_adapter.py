import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _unit_to_years(amount: float, unit: str) -> float:
    u = str(unit).strip().upper()
    a = float(amount)
    if u == "D":
        return a / 365.0
    if u == "W":
        return a * 7.0 / 365.0
    if u == "M":
        return a / 12.0
    if u == "Y":
        return a
    raise ValueError(f"Unsupported term unit '{unit}'")


def _terms_dict_to_years(terms_dict: Dict[str, Sequence]) -> np.ndarray:
    amounts = terms_dict.get("Amount", [])
    units = terms_dict.get("Unit", [])
    if len(amounts) != len(units):
        raise ValueError("terms Amount/Unit lengths mismatch")
    years = np.array([_unit_to_years(a, u) for a, u in zip(amounts, units)], dtype=float)
    return years


def _extract_spot(model_dict: Dict) -> float:
    series = model_dict.get("spot_price_series", {})
    vals = series.get("values", [])
    if not vals:
        raise ValueError(f"Missing spot_price_series.values for model '{model_dict.get('name', 'unknown')}'")
    return float(vals[-1])


def _find_yield_curve_by_currency(yield_curves: Dict[str, Dict], currency: str) -> Tuple[str, Dict]:
    cc = str(currency).upper()
    if not cc:
        raise ValueError("Asset currency is missing; cannot select risk-free curve by currency")

    candidates: List[Tuple[str, Dict]] = []
    for name, curve in yield_curves.items():
        n = str(name).upper()
        cn = str(curve.get("name", "")).upper()
        if cc in n or cc in cn:
            candidates.append((name, curve))

    if not candidates:
        raise ValueError(f"No risk-free yield curve found for currency '{cc}'")
    if len(candidates) == 1:
        return candidates[0]

    # Prefer exact terminal match by naming convention (e.g., YieldCurveEUR).
    for name, curve in candidates:
        n = str(name).upper()
        cn = str(curve.get("name", "")).upper()
        if n.endswith(cc) or cn.endswith(cc):
            return name, curve
    # Deterministic fallback.
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _curve_to_engine_array(curve_dict: Dict) -> np.ndarray:
    rates = np.asarray(curve_dict.get("rates", []), dtype=float)
    years = _terms_dict_to_years(curve_dict.get("terms", {}))
    if rates.size != years.size:
        raise ValueError("Yield curve rates length does not match terms length")
    order = np.argsort(years)
    years = years[order]
    rates = rates[order]
    return np.column_stack([years, rates])


def _dividend_curve_to_engine_array(model_dict: Dict) -> np.ndarray:
    rates = np.asarray(model_dict.get("yield_rates", []), dtype=float)
    years = _terms_dict_to_years(model_dict.get("yield_terms", {}))
    if rates.size != years.size:
        raise ValueError("Dividend curve yield_rates length does not match yield_terms length")
    order = np.argsort(years)
    years = years[order]
    rates = rates[order]
    return np.column_stack([years, rates])


def _reshape_surface_from_flat(
    flat_vols: Sequence[float],
    flat_strikes: Sequence[float],
    flat_terms: Dict[str, Sequence],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vols = np.asarray(flat_vols, dtype=float).reshape(-1)
    strikes = np.asarray(flat_strikes, dtype=float).reshape(-1)
    terms_years = _terms_dict_to_years(flat_terms).reshape(-1)

    n = vols.size
    if strikes.size != n or terms_years.size != n:
        raise ValueError("Surface vectors (vols/strikes/terms) must have same length")

    # Expected UnRisk layout: strike-major blocks, each block cycles all tenors.
    first_strike = strikes[0]
    n_terms = int(np.sum(strikes == first_strike))
    if n_terms <= 0 or n % n_terms != 0:
        raise ValueError("Could not infer (n_terms, n_strikes) layout from surface")
    n_strikes = n // n_terms

    strikes_2d = strikes.reshape(n_strikes, n_terms)
    terms_2d = terms_years.reshape(n_strikes, n_terms)
    vols_2d = vols.reshape(n_strikes, n_terms)

    if not np.allclose(terms_2d, terms_2d[0:1, :], atol=0.0, rtol=0.0):
        raise ValueError("Term grid is not consistent across strike blocks")
    if not np.allclose(strikes_2d, strikes_2d[:, 0:1], atol=0.0, rtol=0.0):
        raise ValueError("Strike blocks are not constant by row")

    tenor_vec = terms_2d[0, :]
    strike_vec = strikes_2d[:, 0]
    vol_matrix = vols_2d.T  # (n_terms, n_strikes), aligned with engine expected shape.

    order_t = np.argsort(tenor_vec)
    order_k = np.argsort(strike_vec)
    tenor_vec = tenor_vec[order_t]
    strike_vec = strike_vec[order_k]
    vol_matrix = vol_matrix[order_t, :][:, order_k]

    return tenor_vec, strike_vec, vol_matrix


def _load_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_assets(
    equity_models: Dict[str, Dict],
    asset_names: Optional[Iterable[str]],
    asset_index: Optional[int],
) -> List[str]:
    available = list(equity_models.keys())
    if asset_index is not None:
        idx = int(asset_index)
        if idx < 0 or idx >= len(available):
            raise IndexError(f"asset_index {idx} is out of range [0, {len(available)-1}]")
        return [available[idx]]
    if asset_names is None:
        return available
    selected = [name for name in asset_names]
    missing = [x for x in selected if x not in equity_models]
    if missing:
        raise ValueError(f"Requested assets not found in file: {missing}")
    return selected


def _build_payload(
    path: str,
    asset_names: Optional[Iterable[str]],
    asset_index: Optional[int],
    surface_kind: str,
) -> Dict:
    data = _load_json(path)
    md = data.get("market_data", {})
    models = md.get("models", {})
    equity_models = models.get("equity_models", {})
    yield_curves = md.get("yield_curves", {})

    if not isinstance(equity_models, dict) or not equity_models:
        raise ValueError("No market_data.models.equity_models found")
    if not isinstance(yield_curves, dict) or not yield_curves:
        raise ValueError("No market_data.yield_curves found")
    if surface_kind not in {"implied", "local"}:
        raise ValueError("surface_kind must be 'implied' or 'local'")

    names = _select_assets(equity_models, asset_names, asset_index)

    s0_list: List[float] = []
    r_zero_list: List[np.ndarray] = []
    q_zero_list: List[np.ndarray] = []
    vol_matrices: List[np.ndarray] = []
    strikes_list: List[np.ndarray] = []
    tenors_list: List[np.ndarray] = []
    currency_list: List[str] = []
    yield_curve_names: List[str] = []

    for name in names:
        model = equity_models[name]
        currency = str(model.get("currency", "")).upper()
        yc_name, yc = _find_yield_curve_by_currency(yield_curves, currency)

        s0_list.append(_extract_spot(model))
        r_zero_list.append(_curve_to_engine_array(yc))
        q_zero_list.append(_dividend_curve_to_engine_array(model))
        currency_list.append(currency)
        yield_curve_names.append(yc_name)

        if surface_kind == "implied":
            surf = model.get("implied_volatility_surface", {})
            t, k, m = _reshape_surface_from_flat(
                flat_vols=surf.get("implied_volatilities", []),
                flat_strikes=surf.get("implied_volatility_strikes", []),
                flat_terms=surf.get("implied_volatility_terms", {}),
            )
        else:
            surf = model.get("local_volatility_surface", {})
            t, k, m = _reshape_surface_from_flat(
                flat_vols=surf.get("local_volatilities", []),
                flat_strikes=surf.get("local_volatility_strikes", []),
                flat_terms=surf.get("local_volatility_terms", {}),
            )

        tenors_list.append(t)
        strikes_list.append(k)
        vol_matrices.append(m)

    return {
        "source_path": str(path),
        "asset_names": names,
        "n_assets": len(names),
        "currency_list": currency_list,
        "yield_curve_names": yield_curve_names,
        "S0_list": s0_list,
        "r_zero_list": r_zero_list,
        "q_zero_list": q_zero_list,
        "vol_matrices": vol_matrices,
        "strikes_list": strikes_list,
        "tenors_list": tenors_list,
    }


def list_equity_assets(path: str) -> List[Dict]:
    data = _load_json(path)
    equity_models = data.get("market_data", {}).get("models", {}).get("equity_models", {})
    if not isinstance(equity_models, dict) or not equity_models:
        raise ValueError("No market_data.models.equity_models found")
    out: List[Dict] = []
    for idx, (name, model) in enumerate(equity_models.items()):
        out.append({
            "index": idx,
            "name": name,
            "currency": str(model.get("currency", "")),
        })
    return out


def read_implied_for_engine(
    path: str,
    asset_names: Optional[Iterable[str]] = None,
    asset_index: Optional[int] = None,
) -> Dict:
    """
    Read UnRisk JSON and return implied-vol payload ready for DupireSimulator.calibrate(...).

    Returns:
        dict with keys:
          - asset_names, n_assets
          - S0_list, r_zero_list, q_zero_list
          - vol_matrices, strikes_list, tenors_list
    """
    return _build_payload(path=path, asset_names=asset_names, asset_index=asset_index, surface_kind="implied")


def read_local_for_engine(
    path: str,
    asset_names: Optional[Iterable[str]] = None,
    asset_index: Optional[int] = None,
) -> Dict:
    """
    Read UnRisk JSON and return local-vol payload ready to value directly with DupireSimulator
    (by building with vol_matrices/tenor_grids/strike_grids) or to set calibrated surfaces.
    """
    return _build_payload(path=path, asset_names=asset_names, asset_index=asset_index, surface_kind="local")


def engine_local_to_unrisk_surface(
    tenors: Sequence[float],
    strikes: Sequence[float],
    local_vol_matrix: Sequence[Sequence[float]],
    term_unit: str = "D",
) -> Dict:
    """
    Convert engine local-vol surface (shape (nT, nK)) to UnRisk JSON layout:
      {
        "local_volatilities": [...],
        "local_volatility_strikes": [...],
        "local_volatility_terms": {"Amount":[...], "Unit":[...]}
      }
    using strike-major flattening with tenors cycling inside each strike block.
    """
    t = np.asarray(tenors, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)
    lv = np.asarray(local_vol_matrix, dtype=float)
    if lv.shape != (t.size, k.size):
        raise ValueError(f"local_vol_matrix shape {lv.shape} does not match ({t.size}, {k.size})")
    if np.any(~np.isfinite(lv)):
        raise ValueError("local_vol_matrix contains NaN/Inf")
    if np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("tenors must be strictly positive and increasing")
    if np.any(k <= 0.0) or np.any(np.diff(k) <= 0.0):
        raise ValueError("strikes must be strictly positive and increasing")

    unit = str(term_unit).upper().strip()
    if unit != "D":
        raise ValueError("Only term_unit='D' is supported for UnRisk export in this adapter")

    term_amounts = [int(round(float(x) * 365.0)) for x in t]
    out_vols: List[float] = []
    out_strikes: List[float] = []
    out_terms: List[int] = []

    # Strike-major layout: for each strike, cycle all tenors.
    for j, kj in enumerate(k):
        for i, ti in enumerate(t):
            out_vols.append(float(lv[i, j]))
            out_strikes.append(float(kj))
            out_terms.append(term_amounts[i])

    return {
        "local_volatilities": out_vols,
        "local_volatility_strikes": out_strikes,
        "local_volatility_terms": {
            "Amount": out_terms,
            "Unit": [unit] * len(out_terms),
        },
    }


def write_engine_local_to_unrisk_json(
    path: str,
    tenors: Sequence[float],
    strikes: Sequence[float],
    local_vol_matrix: Sequence[Sequence[float]],
    *,
    asset_index: int,
    out_path: Optional[str] = None,
) -> str:
    """
    Inject engine local-vol surface into UnRisk JSON for one equity asset selected by index.
    Returns output path written.
    """
    data = _load_json(path)
    equity_models = data.get("market_data", {}).get("models", {}).get("equity_models", {})
    if not isinstance(equity_models, dict) or not equity_models:
        raise ValueError("No market_data.models.equity_models found")
    names = list(equity_models.keys())
    idx = int(asset_index)
    if idx < 0 or idx >= len(names):
        raise IndexError(f"asset_index {idx} is out of range [0, {len(names)-1}]")

    asset_name = names[idx]
    surface = engine_local_to_unrisk_surface(tenors=tenors, strikes=strikes, local_vol_matrix=local_vol_matrix)
    equity_models[asset_name]["local_volatility_surface"] = surface

    target = Path(out_path) if out_path is not None else Path(path).with_name(Path(path).stem + "_with_local.json")
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(target)
