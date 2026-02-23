# MCPricing

Equity-derivatives pricing library with:
- Monte Carlo pricing
- Tree pricing (Jarrow-Rudd, constant volatility)
- Dupire local-vol calibration + Monte Carlo valuation

This README mirrors the structure of `Library_Explainer.ipynb`.

## 1) Capabilities

### Products
- European
- Asian
- Digital
- Barrier
- American
- Bermudan
- Autocall / structured payoffs

### Models
- `BS`: Black-Scholes closed form and/or BS Monte Carlo (depending on product)
- `JarrowRuddTree`: binomial tree pricing for tree-compatible products
- `Dupire`: local-vol calibration + Monte Carlo

### Structured products
Path-dependent products are handled through payoff cashflow logic (`cash_flows`), including coupon accumulation and early redemption in autocalls.

## 2) Mathematical techniques

### Monte Carlo pricing
Risk-neutral discounted expectation of cashflows:

$
V_0 = \mathbb{E}^{\mathbb{Q}}\left[\sum_i D(0,t_i)\,CF_i\right]
$

Implementation flow:
1. Simulator generates paths on observation dates.
2. Payoff maps each path into cashflows.
3. Instrument discounts and averages across simulations.

### Tree pricing (constant vol)
Backward induction on a Jarrow-Rudd tree:
1. Terminal payoff at maturity.
2. Discounted continuation values backwards in time.
3. Early-exercise check (`max(intrinsic, continuation)`) at decision dates.

### Dupire calibration
`Dupire` pricing path:
1. Input tenor/strike grid and implied-vol (or price) surface.
2. Calibrate local-vol surface.
3. Simulate local-vol dynamics.
4. Discount with the same risk-free zero-curve representation used in simulation.

## 3) Library architecture

### `Payoffs.py`
- Payoff definitions (`value`, `cash_flows`, optional `hasDecision`)
- Path-dependent logic (autocall, barriers, etc.)

### `Instruments.py`
- Generic pricing interfaces (`price`, `priceMC`, `priceTree`)
- Discounting integration
- Shared Dupire calibration helpers for one-strike products

### `EquityInstruments.py`
- Concrete product classes
- Model dispatch per product (`BS`, `JarrowRuddTree`, `Dupire`)

### `SimulatorEngine.py`
- Simulation engines (Black-Scholes, tree, Dupire)
- Dupire calibration/simulation internals

### `Utils.py`
- Interpolation helpers (`Matrix2D`, volatility matrices)
- Utility discounting from zero curves

## 4) Technique comparison examples

### A) MC vs Tree (constant vol)
```python
from EquityInstruments import EuropeanCall

op = EuropeanCall(100, 100, 0.02, 0.01, 0.20, 1.0)
price_bs = op.price(model="BS")
price_mc = op.price(model="BS", simulation=True, n_sims=100000)
price_tree = op.price(model="JarrowRuddTree", deltat=1/252)

print(price_bs, price_mc, price_tree)
```

### B) MC constant vol vs Dupire
```python
import numpy as np
from EquityInstruments import EuropeanCall

tenors = np.array([0.25, 0.50, 1.00, 2.00])
strikes = np.array([80, 90, 100, 110, 120])
Tg, Kg = np.meshgrid(tenors, strikes, indexing="ij")
iv_surface = 0.18 + 0.015*np.sqrt(Tg) - 0.0008*(Kg-100) + 0.00002*(Kg-100)**2
iv_surface = np.maximum(iv_surface, 0.05)

op = EuropeanCall(100, 100, 0.02, 0.01, 0.20, 1.0)
price_dupire = op.price(
    model="Dupire",
    n_sims=100000,
    vol_matrix=iv_surface,
    strikes=strikes,
    tenors=tenors,
    vol_floor=1e-8,
)
print(price_dupire)
```

## 5) Create your own payoff

Minimum contract:
1. Inherit from `Payoff`.
2. Implement `value(observations, **kwargs)`.
3. Implement `cash_flows(observations)` if product pays before maturity.
4. Implement `hasDecision(...)` if used by early-exercise tree pricing.

Example:
```python
import numpy as np
from Payoffs import Payoff

class CouponAndRedemptionPayoff(Payoff):
    def __init__(self, notional, coupon_rate, trigger=1.0):
        self.notional = float(notional)
        self.coupon_rate = float(coupon_rate)
        self.trigger = float(trigger)

    def value(self, observations, **kwargs):
        return self.cash_flows(observations)[-1]

    def cash_flows(self, observations):
        obs = np.asarray(observations, dtype=float)
        cf = np.zeros(obs.size)
        cf += (obs > self.trigger) * (self.notional * self.coupon_rate)
        cf[-1] += self.notional
        return cf
```

Then price it with an existing instrument wrapper and simulator:
```python
from EquityInstruments import EuropeanCall
from SimulatorEngine import BlackScholesSimulator

host = EuropeanCall(100, 100, 0.02, 0.01, 0.20, 1.0)
sim = BlackScholesSimulator(100, 0.02, 0.01, 0.20)
payoff = CouponAndRedemptionPayoff(100, 0.01, trigger=0.95)

price = host.priceMC(sim, payoff, dates=[0.25, 0.5, 0.75, 1.0], n_sims=50000)
print(price)
```

## Reference

For full walkthrough and runnable cells, see:
- `Library_Explainer.ipynb`
