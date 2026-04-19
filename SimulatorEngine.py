from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
from scipy.special import ndtr
from scipy.interpolate import interp1d, BSpline
from scipy.linalg import solve_banded
from scipy.integrate import cumulative_trapezoid
from Utils import LocalVolatilityMatrix

class CalibrationState:
    NOT_CALIBRATED = 'NOT_CALIBRATED'
    CALIBRATED = 'CALIBRATED'
    CALIBRATED_CAP_FLOORLET = 'CALIBRATED_CAP_FLOORLET'
    CALIBRATED_CAP_FLOOR = 'CALIBRATED_CAP_FLOOR'
    CALIBRATED_SWAPTION = 'CALIBRATED_SWAPTION'

#######  Simuladores ########

class Simulator(ABC):
    @abstractmethod
    def __init__(self, n_vars, correlMatrix=None):
        super().__init__()
        self.n_vars_ = n_vars
        self.correlMatrix_ = correlMatrix
        if n_vars > 1 and correlMatrix is None:
            raise ValueError("If simulator needs to simultae more than one variable, a correlation matrix is needed")
        
        #TODO: heston has 2 vars, vol and underlying

    def simulate(self, dates, n_sims):
        #TODO: Dates should be a list of positive floats
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.append(deltas, dates[0])
        else:
            deltas = np.array(dates)
        
        if self.n_vars_ == 1:
            return self.simulateWithNoise(dates, n_sims, np.random.standard_normal(  (n_sims, len(deltas))))
        
        elif self.n_vars_ > 1:
            L = np.linalg.cholesky(self.correlMatrix_)
            Z = np.random.standard_normal((n_sims, len(deltas), self.n_vars_))
            Z_correlated = Z @ L.T
            return self.simulateWithNoise(dates, n_sims, Z_correlated)
    
    @abstractmethod
    def simulateWithNoise(self, dates, n_sims, noise):
        pass

class ShiftedLognormalSimulator(Simulator):
    def __init__(self, S0, r, q, sigma, shift, n_assets=1, correlMatrix=None):
        super().__init__(n_vars=n_assets, correlMatrix=correlMatrix)
        if S0 + shift < 0:
            #TODO: lanzar excepción
            pass 

        if n_assets > 1 and correlMatrix is None:
            pass #TODO: lanzar excepción
    
        if n_assets == 1:
            self.S0_ = S0
            self.r_ = r
            self.q_ = q
            self.sigma_ = sigma
            self.shift_ = shift
        else:
            self.S0_ = np.array(S0)
            self.r_ = np.array(r)
            self.q_ = np.array(q)
            self.sigma_ = np.array(sigma)
            self.shift_ = np.array(shift)

    def simulateWithNoise(self, dates, n_sims, noise):
        #TODO: check noise is in shape n_sims x dates
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.append(deltas, dates[0])
        else:
            deltas = np.array(dates)

        if self.n_vars_ == 1:
            lognorm = np.exp( (self.r_ - self.q_- self.sigma_**2/2)*deltas+ np.multiply( noise, np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_)
            observations = (self.S0_+self.shift_) * np.cumprod( lognorm, axis=1 ) - self.shift_
            return observations
        else:
            sigma = self.sigma_.reshape(1, 1, -1)
            r = self.r_.reshape(1, 1, -1)
            q = self.q_.reshape(1, 1, -1)
            S0 = self.S0_.reshape(1, 1, -1)
            shift = self.shift_.reshape(1, 1, -1)
            deltas = deltas.reshape(1, -1, 1)
            sqrt_deltas = np.sqrt(deltas)

            drifts = (r - q - 0.5 * sigma ** 2) * deltas
            diffusion = noise * sigma * sqrt_deltas
            lognorm = np.exp(drifts + diffusion)

            observations = (S0 + shift) * np.cumprod(lognorm, axis=1) - shift
            return observations
    
class BlackScholesSimulator(ShiftedLognormalSimulator):
    def __init__(self, S0, r, q, sigma, n_assets=1):
        super().__init__(S0, r, q, sigma, shift=0, n_assets=n_assets)

class Black76Simulator(BlackScholesSimulator):
    def __init__(self, S0, sigma):
        super().__init__(S0, 0, 0, sigma)

class BachelierSimulator(Simulator):
    def __init__(self, S0, r, q, sigma, n_assets=1):
        super().__init__(n_vars=n_assets)
        if n_assets == 1:
            self.S0_ = S0
            self.r_ = r
            self.q_ = q
            self.sigma_ = sigma
        else:
            self.S0_ = np.array(S0)
            self.r_ = np.array(r)
            self.q_ = np.array(q)
            self.sigma_ = np.array(sigma)
    
    def simulateWithNoise(self, dates, n_sims, noise):
        if len(dates) > 1:
            deltas = np.array(dates[1:])-np.array(dates[:-1])
            deltas = np.insert(deltas, 0, dates[0])
        else:
            deltas = np.array(dates)
        
        if self.n_vars_ == 1:
            norm = (self.r_ - self.q_)*deltas+ np.multiply(noise, np.tile(np.sqrt(deltas), (n_sims,1)))*self.sigma_
            observations = self.S0_ + np.cumsum( norm, axis=1 )

            return observations 
        else:
            if noise.shape != (n_sims, len(dates), self.n_vars_):
                #TODO: lanzar error en ingles
                raise ValueError(f"Shape esperado del ruido: ({n_sims}, {len(dates)}, {self.n_vars_}), recibido: {noise.shape}")
            
            sigma = self.sigma_.reshape(1, 1, -1)
            r = self.r_.reshape(1, 1, -1)
            q = self.q_.reshape(1, 1, -1)
            S0 = self.S0_.reshape(1, 1, -1)

            deltas = deltas.reshape(1, -1, 1)  # shape (1, T, 1)
            sqrt_deltas = np.sqrt(deltas)     # shape (1, T, 1)

            norm = (r - q) * deltas + noise * sigma * sqrt_deltas

            observations = S0 + np.cumsum(norm, axis=1)
            return observations
    
class DupireSimulator(Simulator):
    def __init__(self, S0, r_zero, q_zero, vol_matrices=None, tenor_grids=None, strike_grids=None, n_assets=1, correlMatrix=None):
        # TODO: check if S0 , etc are lists of the same length as n_assets

        super().__init__(n_vars=n_assets, correlMatrix=correlMatrix)

        self.n_assets = n_assets
        self.S0_ = np.array(S0, dtype=float) if n_assets > 1 else float(S0)
        self.state_ = CalibrationState.NOT_CALIBRATED
        self.local_vol_matrices_ = []
        self.vol_interpolators_ = []
        self.r_interps_ = []
        self.q_interps_ = []
        self.r_zero_interps_ = []
        self.q_zero_interps_ = []

        r_inputs = self._to_asset_list(r_zero, n_assets)
        q_inputs = self._to_asset_list(q_zero, n_assets)

        for i in range(n_assets):
            r_zero_interp = self._build_rate_interpolator(r_inputs[i])
            q_zero_interp = self._build_rate_interpolator(q_inputs[i])
            self.r_zero_interps_.append(r_zero_interp)
            self.q_zero_interps_.append(q_zero_interp)
            self.r_interps_.append(self._build_forward_from_zero(r_zero_interp))
            self.q_interps_.append(self._build_forward_from_zero(q_zero_interp))

        if vol_matrices is not None and tenor_grids is not None and strike_grids is not None:
            vol_inputs = self._to_asset_list(vol_matrices, n_assets)
            tenor_inputs = self._to_asset_list(tenor_grids, n_assets)
            strike_inputs = self._to_asset_list(strike_grids, n_assets)

            for i in range(n_assets):
                self.vol_interpolators_.append(
                    self._build_local_vol_interpolator(tenor_inputs[i], strike_inputs[i], vol_inputs[i])
                )
            self.state_ = CalibrationState.CALIBRATED


    def simulateWithNoise(self, dates, n_sims, noise):
        if self.state_ != CalibrationState.CALIBRATED:
            raise ValueError("Dupire simulator must be calibrated before simulation")

        if len(self.vol_interpolators_) != self.n_assets:
            raise ValueError("Dupire simulator is not calibrated: missing local vol interpolators")

        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or np.any(dates <= 0):
            raise ValueError("dates must be a 1D array-like of strictly positive times")

        if len(dates) > 1:
            deltas = dates[1:] - dates[:-1]
            deltas = np.insert(deltas, 0, dates[0])
        else:
            deltas = np.array(dates, dtype=float)
        sqrt_deltas = np.sqrt(deltas)

        if self.n_vars_ == 1:
            noise = np.asarray(noise, dtype=float)
            if noise.shape != (n_sims, len(dates)):
                raise ValueError(f"Expected noise shape ({n_sims}, {len(dates)}), got {noise.shape}")

            observations = np.empty((n_sims, len(dates)), dtype=float)
            s_prev = np.full(n_sims, float(self.S0_), dtype=float)
            floor = 1e-12
            eval_points = np.empty((n_sims, 2), dtype=float)
            t_prev_path = dates - deltas
            r_path = np.empty(len(dates), dtype=float)
            q_path = np.empty(len(dates), dtype=float)
            for i in range(len(dates)):
                t_prev = t_prev_path[i]
                r_path[i] = float(self.r_interps_[0](t_prev))
                q_path[i] = float(self.q_interps_[0](t_prev))

            for i in range(len(dates)):
                dt = deltas[i]
                t_prev = t_prev_path[i]
                sqrt_dt = sqrt_deltas[i]

                r_t = r_path[i]
                q_t = q_path[i]
                eval_points[:, 0] = t_prev
                eval_points[:, 1] = s_prev
                sigma_loc = np.asarray(self.vol_interpolators_[0](eval_points), dtype=float)
                sigma_loc = np.maximum(sigma_loc, 1e-12)

                expo = (r_t - q_t - 0.5 * sigma_loc**2) * dt + sigma_loc * sqrt_dt * noise[:, i]
                s_next = np.maximum(s_prev * np.exp(expo), floor)
                observations[:, i] = s_next
                s_prev = s_next

            return observations

        noise = np.asarray(noise, dtype=float)
        if noise.shape != (n_sims, len(dates), self.n_vars_):
            raise ValueError(f"Expected noise shape ({n_sims}, {len(dates)}, {self.n_vars_}), got {noise.shape}")

        observations = np.empty((n_sims, len(dates), self.n_vars_), dtype=float)
        s_prev = np.tile(self.S0_.reshape(1, -1), (n_sims, 1))
        floor = 1e-12
        eval_points = np.empty((n_sims, 2), dtype=float)
        t_prev_path = dates - deltas
        r_paths = np.empty((self.n_vars_, len(dates)), dtype=float)
        q_paths = np.empty((self.n_vars_, len(dates)), dtype=float)
        for j in range(self.n_vars_):
            for i in range(len(dates)):
                t_prev = t_prev_path[i]
                r_paths[j, i] = float(self.r_interps_[j](t_prev))
                q_paths[j, i] = float(self.q_interps_[j](t_prev))

        for i in range(len(dates)):
            dt = deltas[i]
            t_prev = t_prev_path[i]
            sqrt_dt = sqrt_deltas[i]

            for j in range(self.n_vars_):
                r_t = r_paths[j, i]
                q_t = q_paths[j, i]

                eval_points[:, 0] = t_prev
                eval_points[:, 1] = s_prev[:, j]
                sigma_loc = np.asarray(self.vol_interpolators_[j](eval_points), dtype=float)
                sigma_loc = np.maximum(sigma_loc, 1e-12)

                expo = (r_t - q_t - 0.5 * sigma_loc**2) * dt + sigma_loc * sqrt_dt * noise[:, i, j]
                s_prev[:, j] = np.maximum(s_prev[:, j] * np.exp(expo), floor)

            observations[:, i, :] = s_prev

        return observations

    def _build_rate_interpolator(self, rate_data):
        if callable(rate_data):
            return rate_data
        if np.isscalar(rate_data):
            c = float(rate_data)
            return lambda t, c=c: c

        if isinstance(rate_data, dict):
            ts = np.array(sorted(rate_data.keys()), dtype=float)
            rs = np.array([rate_data[t] for t in ts], dtype=float)
            if ts.size == 1:
                c = float(rs[0])
                return lambda t, c=c: c
            return interp1d(ts, rs, kind='linear', fill_value='extrapolate', bounds_error=False)

        arr = np.asarray(rate_data, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            ts = arr[:, 0]
            rs = arr[:, 1]
            return interp1d(ts, rs, kind='linear', fill_value='extrapolate', bounds_error=False)

        raise ValueError("Rate curve must be callable, scalar, dict, or array-like with shape (n, 2)")

    def _build_local_vol_interpolator(self, tenors, strikes, vol_matrix):
        if isinstance(vol_matrix, LocalVolatilityMatrix):
            local_matrix = vol_matrix
        else:
            local_matrix = LocalVolatilityMatrix(tenors, strikes, vol_matrix)

        self.local_vol_matrices_.append(local_matrix)
        return local_matrix.get_interpolator()

    def _build_forward_from_zero(self, zero_interp):
        # f(t) = d/dt [t * z(t)] = z(t) + t * z'(t)
        def _forward(t):
            t_arr = np.asarray(t, dtype=float)
            eps = np.maximum(1e-5, 1e-4 * np.maximum(t_arr, 1.0))
            t_up = t_arr + eps
            t_dn = np.maximum(t_arr - eps, 1e-12)

            z_t = np.asarray(zero_interp(t_arr), dtype=float)
            z_up = np.asarray(zero_interp(t_up), dtype=float)
            z_dn = np.asarray(zero_interp(t_dn), dtype=float)
            dz_dt = (z_up - z_dn) / (t_up - t_dn)
            return z_t + t_arr * dz_dt

        return _forward

    def _to_asset_list(self, data, n_assets):
        if n_assets == 1:
            if isinstance(data, (list, tuple)) and len(data) == 1:
                return [data[0]]
            return [data]

        if not isinstance(data, (list, tuple)) or len(data) != n_assets:
            raise ValueError(f"Expected list/tuple with {n_assets} entries")
        return list(data)

    def _black_scholes_call_price_and_vega(self, s0, k_grid, t_eff, sqrt_t, zr, zq, df_r, df_q, log_m, sigma):
        vol_t = sigma * sqrt_t
        d1 = (log_m + (zr - zq + 0.5 * sigma * sigma) * t_eff) / np.maximum(vol_t, 1e-12)
        d2 = d1 - vol_t

        nd1 = ndtr(d1)
        nd2 = ndtr(d2)
        call = s0 * df_q * nd1 - k_grid * df_r * nd2

        pdf_d1 = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        vega = s0 * df_q * sqrt_t * pdf_d1
        return call, vega

    def _calibrate_local_vol_surface(self, market_prices, s0, tenors, strikes, r_zero_interp, q_zero_interp,
                                     initial_sigma, vol_floor, *,
                                     smooth_w=0.001, lr=0.07, max_iter=250, tol=1e-8):
        t_grid, k_grid = np.meshgrid(np.asarray(tenors, dtype=float), np.asarray(strikes, dtype=float), indexing="ij")
        t_eff = np.maximum(t_grid, 1e-12)
        sqrt_t = np.sqrt(t_eff)
        zr = np.asarray(r_zero_interp(t_eff), dtype=float)
        zq = np.asarray(q_zero_interp(t_eff), dtype=float)
        df_r = np.exp(-zr * t_eff)
        df_q = np.exp(-zq * t_eff)
        log_m = np.log(np.maximum(s0, 1e-12) / np.maximum(k_grid, 1e-12))
        target = np.asarray(market_prices, dtype=float)
        sigma = np.maximum(np.asarray(initial_sigma, dtype=float), vol_floor)

        n_t, n_k = sigma.shape
        n_total = float(n_t * n_k)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        max_iter = int(max_iter)
        smooth_w = float(smooth_w)
        lr = float(lr)
        tol = float(tol)
        m = np.zeros_like(sigma)
        v = np.zeros_like(sigma)
        prev_loss = np.inf

        for it in range(1, max_iter + 1):
            model_prices, vega = self._black_scholes_call_price_and_vega(
                s0, k_grid, t_eff, sqrt_t, zr, zq, df_r, df_q, log_m, sigma
            )
            err = model_prices - target
            data_loss = 0.5 * np.mean(err * err)

            grad = (err * vega) / n_total

            smooth_loss = 0.0
            if n_t > 1:
                d_t = sigma[1:, :] - sigma[:-1, :]
                smooth_loss += 0.5 * smooth_w * np.mean(d_t * d_t)
                grad_t = (smooth_w / d_t.size) * d_t
                grad[1:, :] += grad_t
                grad[:-1, :] -= grad_t

            if n_k > 1:
                d_k = sigma[:, 1:] - sigma[:, :-1]
                smooth_loss += 0.5 * smooth_w * np.mean(d_k * d_k)
                grad_k = (smooth_w / d_k.size) * d_k
                grad[:, 1:] += grad_k
                grad[:, :-1] -= grad_k

            loss = data_loss + smooth_loss
            if np.isfinite(prev_loss):
                rel_impr = abs(prev_loss - loss) / max(abs(prev_loss), 1.0)
                if rel_impr <= tol:
                    break
            prev_loss = loss

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** it)
            v_hat = v / (1.0 - beta2 ** it)

            sigma = sigma - lr * m_hat / (np.sqrt(v_hat) + eps)
            sigma = np.maximum(sigma, vol_floor)

        return sigma

    def _integrate_curve(self, curve, tenors):
        tenors = np.asarray(tenors, dtype=float)
        if tenors.ndim != 1 or tenors.size == 0:
            raise ValueError("tenors must be a non-empty 1D array")

        if curve is None:
            return np.zeros_like(tenors, dtype=float)

        if np.isscalar(curve):
            return float(curve) * tenors

        if callable(curve):
            out = np.empty_like(tenors, dtype=float)
            for j, t in enumerate(tenors):
                if t <= 0.0:
                    out[j] = 0.0
                    continue
                n_steps = max(32, int(np.ceil(64 * t)))
                grid = np.linspace(0.0, t, n_steps + 1)
                rates = np.asarray(curve(grid), dtype=float)
                if rates.shape != grid.shape:
                    rates = np.full_like(grid, float(curve(float(t))), dtype=float)
                out[j] = np.trapz(rates, grid)
            return out

        if isinstance(curve, dict):
            ts = np.array(sorted(curve.keys()), dtype=float)
            rs = np.array([curve[x] for x in ts], dtype=float)
        else:
            arr = np.asarray(curve, dtype=float)
            if arr.ndim == 1:
                if arr.size != tenors.size:
                    raise ValueError("1D rate array must have same length as tenors")
                ts = tenors.copy()
                rs = arr
            elif arr.ndim == 2 and arr.shape[1] == 2:
                ts = arr[:, 0]
                rs = arr[:, 1]
            else:
                raise ValueError("curve must be scalar, callable, dict, 1D array aligned to tenors, or (n,2) array")

        order = np.argsort(ts)
        ts = ts[order]
        rs = rs[order]
        if ts[0] > 0.0:
            ts = np.insert(ts, 0, 0.0)
            rs = np.insert(rs, 0, rs[0])
        elif ts[0] < 0.0:
            raise ValueError("curve times must be >= 0")

        r_interp = interp1d(ts, rs, kind='linear', fill_value='extrapolate', bounds_error=False)
        out = np.empty_like(tenors, dtype=float)
        for j, t in enumerate(tenors):
            if t <= 0.0:
                out[j] = 0.0
                continue
            n_steps = max(16, int(np.ceil(32 * t)))
            grid = np.linspace(0.0, t, n_steps + 1)
            out[j] = np.trapz(np.asarray(r_interp(grid), dtype=float), grid)
        return out

    def _bs_call_from_implied(self, s0, strikes, tenors, implied_vols, int_r, int_q, vol_floor):
        strikes = np.asarray(strikes, dtype=float)
        tenors = np.asarray(tenors, dtype=float)
        implied_vols = np.maximum(np.asarray(implied_vols, dtype=float), vol_floor)
        int_r = np.asarray(int_r, dtype=float)
        int_q = np.asarray(int_q, dtype=float)

        t_grid, k_grid = np.meshgrid(tenors, strikes, indexing="ij")
        int_r_grid = np.repeat(int_r[:, None], strikes.size, axis=1)
        int_q_grid = np.repeat(int_q[:, None], strikes.size, axis=1)
        df_r = np.exp(-int_r_grid)
        df_q = np.exp(-int_q_grid)
        fwd = np.maximum(s0 * df_q / np.maximum(df_r, 1e-16), 1e-16)

        t_eff = np.maximum(t_grid, 1e-12)
        sigma = np.maximum(implied_vols, vol_floor)
        var = np.maximum(sigma * sigma * t_eff, 1e-16)
        std = np.sqrt(var)
        log_fk = np.log(np.maximum(fwd, 1e-16) / np.maximum(k_grid, 1e-16))
        d1 = (log_fk + 0.5 * var) / std
        d2 = d1 - std
        calls = df_r * (fwd * ndtr(d1) - k_grid * ndtr(d2))

        near_zero = t_grid <= 1e-10
        if np.any(near_zero):
            calls[near_zero] = np.maximum(s0 - k_grid[near_zero], 0.0)
        return calls

    def _bs_call_forward(self, F, K, T, vol, DF=1.0):
        F = float(max(F, 1e-16))
        K = float(max(K, 1e-16))
        T = float(max(T, 1e-12))
        vol = float(max(vol, 1e-12))
        DF = float(max(DF, 0.0))
        std = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * std * std) / std
        d2 = d1 - std
        return DF * (F * ndtr(d1) - K * ndtr(d2))

    def _vega_bs(self, F, K, T, vol, DF=1.0):
        F = float(max(F, 1e-16))
        K = float(max(K, 1e-16))
        T = float(max(T, 1e-12))
        vol = float(max(vol, 1e-12))
        DF = float(max(DF, 0.0))
        std = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * std * std) / std
        phi = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
        return DF * F * np.sqrt(T) * phi

    def _implied_vol_from_call_price(self, price, F, K, T, DF, vol_bounds=(1e-8, 5.0), vol_floor=1e-8):
        F = float(max(F, 1e-16))
        K = float(max(K, 1e-16))
        T = float(max(T, 1e-12))
        DF = float(max(DF, 1e-16))
        lo, hi = float(vol_bounds[0]), float(vol_bounds[1])
        intrinsic = max(DF * (F - K), 0.0)
        upper = DF * F
        p = float(np.clip(price, intrinsic, upper))

        def f(sig):
            return self._bs_call_forward(F=F, K=K, T=T, vol=sig, DF=DF) - p

        flo = f(lo)
        fhi = f(hi)
        if flo == 0.0:
            return lo, False
        if fhi == 0.0:
            return hi, False
        if flo * fhi > 0.0:
            # Saturation/failure fallback.
            return (lo if abs(flo) < abs(fhi) else hi), True

        try:
            iv = float(opt.brentq(f, lo, hi, maxiter=200, xtol=1e-12))
            return max(iv, vol_floor), False
        except Exception:
            return (lo if abs(flo) < abs(fhi) else hi), True

    def _implied_vol_surface_from_prices(self, call_prices, strikes, tenors, F_t, DF_r_t, *,
                                         vol_bounds=(1e-8, 5.0), vol_floor=1e-8):
        call_prices = np.asarray(call_prices, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        tenors = np.asarray(tenors, dtype=float)
        F_t = np.asarray(F_t, dtype=float)
        DF_r_t = np.asarray(DF_r_t, dtype=float)
        if call_prices.shape != (tenors.size, strikes.size):
            raise ValueError("call_prices shape mismatch for implied-vol inversion")

        out = np.empty_like(call_prices, dtype=float)
        n_fail = 0
        for i, t in enumerate(tenors):
            F = float(F_t[i])
            DF = float(DF_r_t[i])
            for j, k in enumerate(strikes):
                iv, failed = self._implied_vol_from_call_price(
                    price=float(call_prices[i, j]), F=F, K=float(k), T=float(t), DF=DF,
                    vol_bounds=vol_bounds, vol_floor=vol_floor
                )
                out[i, j] = iv
                n_fail += int(failed)
        return out, n_fail

    def _prepare_iv_weight_matrix(self, weights, iv_mkt, strikes, tenors, F_t, DF_r_t):
        iv_mkt = np.asarray(iv_mkt, dtype=float)
        shape = iv_mkt.shape
        strikes = np.asarray(strikes, dtype=float)
        tenors = np.asarray(tenors, dtype=float)
        F_t = np.asarray(F_t, dtype=float)
        DF_r_t = np.asarray(DF_r_t, dtype=float)

        if weights is None:
            return np.ones(shape, dtype=float)

        if isinstance(weights, str):
            mode = weights.strip().lower()
            if mode == "vega":
                w = np.empty(shape, dtype=float)
                for i, t in enumerate(tenors):
                    F = float(F_t[i])
                    DF = float(DF_r_t[i])
                    for j, k in enumerate(strikes):
                        vega = self._vega_bs(F=F, K=float(k), T=float(t), vol=float(max(iv_mkt[i, j], 1e-8)), DF=DF)
                        vega = max(vega, 1e-6)
                        w[i, j] = 1.0 / (vega * vega)
                return w
            if mode == "atm":
                w = np.empty(shape, dtype=float)
                for i, _t in enumerate(tenors):
                    logm = np.log(np.maximum(strikes, 1e-16) / max(float(F_t[i]), 1e-16))
                    w[i, :] = np.exp(-(logm / 0.25) ** 2)
                return np.maximum(w, 1e-8)
            raise ValueError("weights string mode must be one of: 'vega', 'atm'")

        return self._prepare_weight_matrix(weights, shape)

    def _build_iv_residual(self, theta_vec, *, s0, tenors, strikes, iv_mkt, sqrt_w,
                           r_curve, q_curve, int_r, int_q, vol_floor, alpha,
                           d2_t, d2_k,
                           knots_t, knots_k, degree, use_log_sigma,
                           nK_pde=None, nT_pde=None, pde_theta=1.0,
                           return_parts=False):
        tenors = np.asarray(tenors, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        iv_mkt = np.asarray(iv_mkt, dtype=float)
        sqrt_w = np.asarray(sqrt_w, dtype=float)
        int_r = np.asarray(int_r, dtype=float)
        int_q = np.asarray(int_q, dtype=float)

        df_r_t = np.exp(-int_r)
        df_q_t = np.exp(-int_q)
        f_t = np.maximum(float(s0) * df_q_t / np.maximum(df_r_t, 1e-16), 1e-16)

        sigma_fn = self._build_spline_sigma_interpolator(
            theta_vec=theta_vec,
            knots_t=knots_t,
            knots_k=knots_k,
            degree=int(degree),
            use_log_sigma=bool(use_log_sigma),
            vol_floor=vol_floor
        )
        c_model = self._dupire_solve_call_surface(
            S0=s0,
            tenors=tenors,
            strikes=strikes,
            sigma_fn=sigma_fn,
            r_curve=r_curve,
            q_curve=q_curve,
            vol_floor=vol_floor,
            nK_pde=nK_pde,
            nT_pde=nT_pde,
            theta=float(pde_theta),
            spline_fast_eval={
                "theta_vec": np.asarray(theta_vec, dtype=float),
                "knots_t": np.asarray(knots_t, dtype=float),
                "knots_k": np.asarray(knots_k, dtype=float),
                "degree": int(degree),
                "use_log_sigma": bool(use_log_sigma),
            }
        )
        iv_model, n_fail_iv = self._implied_vol_surface_from_prices(
            call_prices=c_model,
            strikes=strikes,
            tenors=tenors,
            F_t=f_t,
            DF_r_t=df_r_t,
            vol_bounds=(1e-8, 5.0),
            vol_floor=vol_floor
        )

        r_price_mat = sqrt_w * (iv_model - iv_mkt)
        r_price = r_price_mat.reshape(-1, order="C")

        n_bt = int(np.asarray(knots_t).size - int(degree) - 1)
        n_bk = int(np.asarray(knots_k).size - int(degree) - 1)
        r_reg, r_reg_t, r_reg_k = self._regularization_residual(
            theta_vec=theta_vec, n_bt=n_bt, n_bk=n_bk, d2_t=d2_t, d2_k=d2_k, alpha=float(alpha)
        )

        r = np.concatenate([r_price, r_reg]) if r_reg.size > 0 else r_price
        if return_parts:
            return r, c_model, iv_model, r_price, r_reg, r_reg_t, r_reg_k, n_fail_iv
        return r, c_model, iv_model, n_fail_iv

    def _make_open_uniform_knots(self, x_min, x_max, degree, n_internal):
        x_min = float(x_min)
        x_max = float(x_max)
        if x_max <= x_min:
            x_max = x_min + 1e-8
        n_internal = int(max(0, n_internal))
        interior = np.linspace(x_min, x_max, n_internal + 2, dtype=float)[1:-1] if n_internal > 0 else np.array([], dtype=float)
        left = np.repeat(x_min, degree + 1)
        right = np.repeat(x_max, degree + 1)
        return np.concatenate([left, interior, right]).astype(float)

    def _bspline_design_matrix(self, x, knots, degree):
        x = np.asarray(x, dtype=float).reshape(-1)
        knots = np.asarray(knots, dtype=float).reshape(-1)
        n_basis = knots.size - degree - 1
        if n_basis <= 0:
            raise ValueError("Invalid knot vector: no basis functions available")

        x_clip = np.clip(x, knots[degree], knots[-degree - 1])
        design = np.zeros((x_clip.size, n_basis), dtype=float)
        eye = np.eye(n_basis, dtype=float)
        for j in range(n_basis):
            spline_j = BSpline(knots, eye[j], degree, extrapolate=True)
            design[:, j] = spline_j(x_clip)
        return design

    def _tensor_bspline_design(self, basis_t, basis_k):
        # Returns Phi where each row corresponds to one (T_i, K_j) grid point.
        return np.einsum("ta,kb->tkab", basis_t, basis_k, optimize=True).reshape(
            basis_t.shape[0] * basis_k.shape[0], basis_t.shape[1] * basis_k.shape[1]
        )

    def _sigma_from_theta_grid(self, theta_vec, basis_t, basis_k, use_log_sigma, vol_floor):
        n_bt = basis_t.shape[1]
        n_bk = basis_k.shape[1]
        theta = np.asarray(theta_vec, dtype=float).reshape(n_bt, n_bk)
        raw = np.einsum("ta,ab,kb->tk", basis_t, theta, basis_k, optimize=True)
        if use_log_sigma:
            sigma = np.exp(raw)
        else:
            sigma = raw
        return np.maximum(sigma, vol_floor)

    def _fit_theta_lsq_from_surface(self, sigma_surface, basis_t, basis_k, use_log_sigma, vol_floor):
        sigma_surface = np.maximum(np.asarray(sigma_surface, dtype=float), vol_floor)
        y = np.log(sigma_surface) if use_log_sigma else sigma_surface
        phi = self._tensor_bspline_design(basis_t, basis_k)
        theta_vec, _, _, _ = np.linalg.lstsq(phi, y.reshape(-1), rcond=None)
        return theta_vec

    def _initialize_theta_from_market(self, market_vols, basis_t, basis_k, strikes, s0, use_log_sigma, vol_floor):
        vols = np.maximum(np.asarray(market_vols, dtype=float), vol_floor)
        strikes = np.asarray(strikes, dtype=float)
        atm_idx = int(np.argmin(np.abs(strikes - float(s0)))) if strikes.size > 0 else 0
        sigma_atm = float(np.nanmean(vols[:, atm_idx])) if vols.shape[1] > 0 else float(np.nanmean(vols))
        sigma_mean = float(np.nanmean(vols))
        sigma0 = sigma_atm if np.isfinite(sigma_atm) and sigma_atm > 0.0 else sigma_mean
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            sigma0 = 0.2
        sigma0 = max(sigma0, vol_floor)

        sigma_init_surface = np.full_like(vols, sigma0, dtype=float)
        try:
            theta_init = self._fit_theta_lsq_from_surface(
                sigma_surface=sigma_init_surface,
                basis_t=basis_t,
                basis_k=basis_k,
                use_log_sigma=bool(use_log_sigma),
                vol_floor=vol_floor
            )
            if np.any(~np.isfinite(theta_init)):
                raise FloatingPointError("non-finite theta_init from LSQ")
        except Exception:
            # Fallback if LSQ projection fails.
            n_bt = basis_t.shape[1]
            n_bk = basis_k.shape[1]
            theta_init = np.zeros(n_bt * n_bk, dtype=float)
            if bool(use_log_sigma):
                theta_init[0] = np.log(sigma0)
            else:
                theta_init[0] = sigma0
        return theta_init, sigma0

    def _build_spline_sigma_interpolator(self, theta_vec, knots_t, knots_k, degree, use_log_sigma, vol_floor):
        """
        Build sigma(T, K) evaluator with API sigma_fn(points) -> array(n,).
        `points` must be shape (n, 2), with columns [T, K].
        Extrapolation behavior: inputs are clipped to spline knot support
        [knots[degree], knots[-degree-1]] in each axis before basis evaluation.
        """
        theta_vec = np.asarray(theta_vec, dtype=float).copy()
        knots_t = np.asarray(knots_t, dtype=float).copy()
        knots_k = np.asarray(knots_k, dtype=float).copy()

        def _interp(points):
            arr = np.asarray(points, dtype=float)
            if arr.ndim == 1:
                if arr.size != 2:
                    raise ValueError("Point input must have exactly 2 entries: [T, K]")
                arr = arr.reshape(1, 2)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Interpolator expects shape (n, 2) with columns [T, K]")

            basis_t_q = self._bspline_design_matrix(arr[:, 0], knots_t, degree)
            basis_k_q = self._bspline_design_matrix(arr[:, 1], knots_k, degree)
            n_bt = basis_t_q.shape[1]
            n_bk = basis_k_q.shape[1]
            theta = theta_vec.reshape(n_bt, n_bk)
            raw = np.einsum("na,ab,nb->n", basis_t_q, theta, basis_k_q, optimize=True)
            if use_log_sigma:
                sigma = np.exp(raw)
            else:
                sigma = raw
            return np.maximum(sigma, vol_floor)

        return _interp

    def _curve_to_callable(self, curve, tenors_ref=None):
        if curve is None:
            return lambda t: np.zeros_like(np.asarray(t, dtype=float), dtype=float)
        if np.isscalar(curve):
            c = float(curve)
            return lambda t, c=c: np.full_like(np.asarray(t, dtype=float), c, dtype=float)
        if callable(curve):
            return curve
        if isinstance(curve, dict):
            ts = np.array(sorted(curve.keys()), dtype=float)
            rs = np.array([curve[x] for x in ts], dtype=float)
            if ts.size == 1:
                c = float(rs[0])
                return lambda t, c=c: np.full_like(np.asarray(t, dtype=float), c, dtype=float)
            i1 = interp1d(ts, rs, kind='linear', fill_value='extrapolate', bounds_error=False)
            return lambda t, i1=i1: np.asarray(i1(np.asarray(t, dtype=float)), dtype=float)

        arr = np.asarray(curve, dtype=float)
        if arr.ndim == 1:
            if tenors_ref is None:
                raise ValueError("1D curve input requires tenors_ref for alignment")
            tr = np.asarray(tenors_ref, dtype=float)
            if arr.size != tr.size:
                raise ValueError("1D curve input length must match tenors_ref length")
            i1 = interp1d(tr, arr, kind='linear', fill_value='extrapolate', bounds_error=False)
            return lambda t, i1=i1: np.asarray(i1(np.asarray(t, dtype=float)), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            i1 = interp1d(arr[:, 0], arr[:, 1], kind='linear', fill_value='extrapolate', bounds_error=False)
            return lambda t, i1=i1: np.asarray(i1(np.asarray(t, dtype=float)), dtype=float)

        raise ValueError("curve must be scalar, callable, dict, 1D aligned array, or (n,2) array")

    def _prepare_weight_matrix(self, weights, target_shape):
        if weights is None:
            return np.ones(target_shape, dtype=float)

        w = np.asarray(weights, dtype=float)
        try:
            w = np.broadcast_to(w, target_shape).astype(float, copy=False)
        except ValueError as exc:
            raise ValueError(f"weights with shape {w.shape} cannot be broadcast to {target_shape}") from exc

        if np.any(~np.isfinite(w)):
            raise ValueError("weights must be finite")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        return w

    def _second_diff_matrix(self, n):
        n = int(n)
        if n <= 2:
            return np.zeros((0, n), dtype=float)
        d2 = np.zeros((n - 2, n), dtype=float)
        idx = np.arange(n - 2)
        d2[idx, idx] = 1.0
        d2[idx, idx + 1] = -2.0
        d2[idx, idx + 2] = 1.0
        return d2

    def _regularization_residual(self, theta_vec, n_bt, n_bk, d2_t, d2_k, alpha):
        if alpha <= 0.0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        theta = np.asarray(theta_vec, dtype=float).reshape(n_bt, n_bk)
        sqrt_alpha = np.sqrt(float(alpha))

        r_t = (d2_t @ theta).reshape(-1, order="C") if d2_t.shape[0] > 0 else np.zeros(0, dtype=float)
        r_k = (theta @ d2_k.T).reshape(-1, order="C") if d2_k.shape[0] > 0 else np.zeros(0, dtype=float)
        r_t *= sqrt_alpha
        r_k *= sqrt_alpha
        r_reg = np.concatenate([r_t, r_k]) if (r_t.size + r_k.size) > 0 else np.zeros(0, dtype=float)
        return r_reg, r_t, r_k

    def _build_reg_jacobian_explicit(self, n_bt, n_bk, d2_t, d2_k, alpha):
        if alpha <= 0.0:
            return np.zeros((0, n_bt * n_bk), dtype=float)

        p = n_bt * n_bk
        n_rt = d2_t.shape[0] * n_bk
        n_rk = n_bt * d2_k.shape[0]
        n_rows = n_rt + n_rk
        if n_rows == 0:
            return np.zeros((0, p), dtype=float)

        sqrt_alpha = np.sqrt(float(alpha))
        j_reg = np.zeros((n_rows, p), dtype=float)
        for k in range(p):
            basis_k = np.zeros((n_bt, n_bk), dtype=float)
            basis_k.reshape(-1, order="C")[k] = 1.0
            part_t = (d2_t @ basis_k).reshape(-1, order="C") if d2_t.shape[0] > 0 else np.zeros(0, dtype=float)
            part_k = (basis_k @ d2_k.T).reshape(-1, order="C") if d2_k.shape[0] > 0 else np.zeros(0, dtype=float)
            j_reg[:, k] = sqrt_alpha * np.concatenate([part_t, part_k])
        return j_reg

    def _build_residual(self, theta_vec, *, s0, tenors, strikes, c_mkt, sqrt_w,
                        r_curve, q_curve, vol_floor, alpha,
                        d2_t, d2_k,
                        knots_t, knots_k, degree, use_log_sigma,
                        nK_pde=None, nT_pde=None, pde_theta=1.0,
                        return_parts=False):
        """
        Build extended residual for LM/Gauss-Newton:
        r(theta) = [r_price; r_reg],  J(theta)=0.5*||r||^2
        where:
          r_price = vec( sqrt(W) * (C_model - C_mkt) ) in C-order
          r_reg   = sqrt(alpha)*[ vec(D2_T @ Theta); vec(Theta @ D2_K^T) ] in C-order
        """
        tenors = np.asarray(tenors, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        c_mkt = np.asarray(c_mkt, dtype=float)
        sqrt_w = np.asarray(sqrt_w, dtype=float)

        if c_mkt.shape != (tenors.size, strikes.size):
            raise ValueError(f"c_mkt shape {c_mkt.shape} does not match ({tenors.size}, {strikes.size})")
        if sqrt_w.shape != c_mkt.shape:
            raise ValueError(f"sqrt_w shape {sqrt_w.shape} does not match c_mkt shape {c_mkt.shape}")

        r_price, c_model = self._build_price_residual(
            theta_vec=theta_vec,
            s0=s0,
            tenors=tenors,
            strikes=strikes,
            c_mkt=c_mkt,
            sqrt_w=sqrt_w,
            r_curve=r_curve,
            q_curve=q_curve,
            vol_floor=vol_floor,
            knots_t=knots_t,
            knots_k=knots_k,
            degree=int(degree),
            use_log_sigma=bool(use_log_sigma),
            nK_pde=nK_pde,
            nT_pde=nT_pde,
            theta=float(pde_theta),
        )

        n_bt = int(np.asarray(knots_t).size - int(degree) - 1)
        n_bk = int(np.asarray(knots_k).size - int(degree) - 1)
        r_reg, r_reg_t, r_reg_k = self._regularization_residual(
            theta_vec=theta_vec, n_bt=n_bt, n_bk=n_bk, d2_t=d2_t, d2_k=d2_k, alpha=float(alpha)
        )

        r = np.concatenate([r_price, r_reg]) if r_reg.size > 0 else r_price
        if return_parts:
            return r, c_model, r_price, r_reg, r_reg_t, r_reg_k
        return r, c_model

    def _build_price_residual(self, theta_vec, *, s0, tenors, strikes, c_mkt, sqrt_w,
                              r_curve, q_curve, vol_floor,
                              knots_t, knots_k, degree, use_log_sigma,
                              nK_pde=None, nT_pde=None, theta=1.0):
        sigma_fn = self._build_spline_sigma_interpolator(
            theta_vec=theta_vec,
            knots_t=knots_t,
            knots_k=knots_k,
            degree=int(degree),
            use_log_sigma=bool(use_log_sigma),
            vol_floor=vol_floor
        )
        c_model = self._dupire_solve_call_surface(
            S0=s0,
            tenors=tenors,
            strikes=strikes,
            sigma_fn=sigma_fn,
            r_curve=r_curve,
            q_curve=q_curve,
            vol_floor=vol_floor,
            nK_pde=nK_pde,
            nT_pde=nT_pde,
            theta=float(theta),
            spline_fast_eval={
                "theta_vec": np.asarray(theta_vec, dtype=float),
                "knots_t": np.asarray(knots_t, dtype=float),
                "knots_k": np.asarray(knots_k, dtype=float),
                "degree": int(degree),
                "use_log_sigma": bool(use_log_sigma),
            }
        )
        if c_model.shape != c_mkt.shape:
            raise ValueError(f"c_model shape {c_model.shape} does not match c_mkt shape {c_mkt.shape}")
        if np.any(~np.isfinite(c_model)):
            raise FloatingPointError("c_model contains NaN/Inf")

        r_price_mat = np.asarray(sqrt_w, dtype=float) * (c_model - np.asarray(c_mkt, dtype=float))
        r_price = r_price_mat.reshape(-1, order="C")
        return r_price, c_model

    def _dupire_solve_call_surface(self, S0, tenors, strikes, sigma_fn, r_curve, q_curve, *,
                                   vol_floor=1e-8, kmax_mult=1.5, nK_pde=None, nT_pde=None,
                                   theta=1.0, monotonicity_tol=1e-7,
                                   spline_fast_eval=None):
        """
        Solve Dupire forward PDE for call prices and return values at market nodes.

        sigma_fn API:
        - Callable with signature sigma_fn(points), where points is ndarray (n, 2)
          with columns [T, K], returning ndarray (n,) of local vols.
        - Values are floored at `vol_floor` inside the solver.

        Extrapolation:
        - The solver queries sigma on K in [0, K_max] and T in [0, T_max].
        - If sigma_fn internally uses spline interpolation from `_build_spline_sigma_interpolator`,
          out-of-support (T, K) are clipped to knot bounds before evaluation.

        Optional fast path:
        - `spline_fast_eval` may be a dict with keys:
          theta_vec, knots_t, knots_k, degree, use_log_sigma.
        - When provided, solver precomputes basis on PDE grids and evaluates sigma
          via `_sigma_from_theta_grid` in block form instead of calling sigma_fn.
        """
        tenors = np.asarray(tenors, dtype=float).reshape(-1)
        strikes = np.asarray(strikes, dtype=float).reshape(-1)
        if tenors.size == 0 or strikes.size == 0:
            raise ValueError("tenors and strikes must be non-empty")
        if np.any(~np.isfinite(tenors)) or np.any(~np.isfinite(strikes)):
            raise ValueError("tenors and strikes must be finite")
        if np.any(tenors <= 0.0):
            raise ValueError("tenors must be strictly positive")
        if np.any(strikes <= 0.0):
            raise ValueError("strikes must be strictly positive")

        nK = int(max(200, 4 * strikes.size) if nK_pde is None else max(int(nK_pde), 4 * strikes.size, 50))
        nT = int(max(200, 10 * tenors.size) if nT_pde is None else max(int(nT_pde), 10 * tenors.size, 50))
        theta = float(theta)
        if theta < 0.5 or theta > 1.0:
            raise ValueError("theta must be in [0.5, 1.0] (0.5=Crank-Nicolson, 1.0=Backward Euler)")

        K_min = 0.0
        K_max = float(np.max(strikes)) * float(kmax_mult)
        if K_max <= np.max(strikes):
            K_max = float(np.max(strikes)) * 1.1
        K_grid = np.linspace(K_min, K_max, nK)

        T_max = float(np.max(tenors))
        T_grid = np.linspace(0.0, T_max, nT)
        if T_grid.size < 2:
            raise ValueError("Time grid must have at least 2 points")
        dt = float(T_grid[1] - T_grid[0])
        dK = float(K_grid[1] - K_grid[0])

        r_fn = self._curve_to_callable(r_curve, tenors_ref=tenors)
        q_fn = self._curve_to_callable(q_curve, tenors_ref=tenors)
        q_vals = np.asarray(q_fn(T_grid), dtype=float)
        if q_vals.shape == ():
            q_vals = np.full_like(T_grid, float(q_vals), dtype=float)
        elif q_vals.shape != T_grid.shape:
            q_vals = np.full_like(T_grid, float(q_fn(float(T_grid[-1]))), dtype=float)
        int_q_grid = cumulative_trapezoid(q_vals, T_grid, initial=0.0)
        left_bc = float(S0) * np.exp(-np.asarray(int_q_grid, dtype=float))
        right_bc = np.zeros_like(T_grid, dtype=float)

        C = np.empty((nT, nK), dtype=float)
        C[0, :] = np.maximum(float(S0) - K_grid, 0.0)
        C[:, 0] = left_bc
        C[:, -1] = right_bc

        K_in = K_grid[1:-1]
        m = K_in.size
        if m < 2:
            raise ValueError("Not enough interior K nodes for PDE solve")

        spline_theta_vec = None
        spline_basis_t_grid = None
        spline_basis_k_grid = None
        spline_use_log = True
        if spline_fast_eval is not None:
            req = ("theta_vec", "knots_t", "knots_k", "degree", "use_log_sigma")
            missing = [k for k in req if k not in spline_fast_eval]
            if missing:
                raise ValueError(f"spline_fast_eval is missing required keys: {missing}")
            spline_theta_vec = np.asarray(spline_fast_eval["theta_vec"], dtype=float)
            spline_basis_t_grid = self._bspline_design_matrix(
                T_grid, np.asarray(spline_fast_eval["knots_t"], dtype=float), int(spline_fast_eval["degree"])
            )
            spline_basis_k_grid = self._bspline_design_matrix(
                K_grid, np.asarray(spline_fast_eval["knots_k"], dtype=float), int(spline_fast_eval["degree"])
            )
            spline_use_log = bool(spline_fast_eval["use_log_sigma"])

        # Semi-implicit theta-scheme with L(T_n) used on both sides:
        # (I - theta*dt*L_n) C^{n+1} = (I + (1-theta)*dt*L_n) C^n + boundary terms.
        for n in range(nT - 1):
            t_n = float(T_grid[n])
            r_n = np.asarray(r_fn(t_n), dtype=float).reshape(-1)[0]
            q_n = np.asarray(q_fn(t_n), dtype=float).reshape(-1)[0]

            if spline_theta_vec is not None:
                sigma_full = self._sigma_from_theta_grid(
                    theta_vec=spline_theta_vec,
                    basis_t=spline_basis_t_grid[n:n + 1, :],
                    basis_k=spline_basis_k_grid,
                    use_log_sigma=spline_use_log,
                    vol_floor=vol_floor
                ).reshape(-1)
            else:
                pts = np.column_stack([np.full(K_grid.size, t_n), K_grid])
                sigma_full = np.asarray(sigma_fn(pts), dtype=float).reshape(-1)
                if sigma_full.size != K_grid.size:
                    raise ValueError("sigma_fn must return one volatility per input point")
                sigma_full = np.maximum(sigma_full, vol_floor)
            sigma_in = sigma_full[1:-1]

            A = 0.5 * sigma_in * sigma_in * (K_in * K_in)
            B = -(r_n - q_n) * K_in
            D = -q_n

            alpha = A / (dK * dK) - B / (2.0 * dK)
            beta = -2.0 * A / (dK * dK) + D
            gamma = A / (dK * dK) + B / (2.0 * dK)

            c_prev = C[n, :]
            rhs = c_prev[1:-1] + (1.0 - theta) * dt * (
                alpha * c_prev[:-2] + beta * c_prev[1:-1] + gamma * c_prev[2:]
            )

            # Boundary adjustment from known C^{n+1} boundaries in the implicit operator.
            rhs[0] += theta * dt * alpha[0] * left_bc[n + 1]
            rhs[-1] += theta * dt * gamma[-1] * right_bc[n + 1]

            lower = -theta * dt * alpha[1:]
            diag = 1.0 - theta * dt * beta
            upper = -theta * dt * gamma[:-1]

            ab = np.zeros((3, m), dtype=float)
            ab[0, 1:] = upper
            ab[1, :] = diag
            ab[2, :-1] = lower
            c_next_in = solve_banded((1, 1), ab, rhs)

            if np.any(~np.isfinite(c_next_in)):
                raise FloatingPointError("Dupire solver produced non-finite values; increase grid density or use theta=1.0")

            C[n + 1, 0] = left_bc[n + 1]
            C[n + 1, -1] = right_bc[n + 1]
            C[n + 1, 1:-1] = c_next_in

        if np.any(~np.isfinite(C)):
            raise FloatingPointError("Dupire call surface contains NaN/Inf")

        dC_dK = np.diff(C, axis=1)
        max_increase = float(np.max(dC_dK))
        if max_increase > monotonicity_tol:
            raise ValueError(
                f"Call surface is not monotone in strike (max dC={max_increase:.3e}); "
                "increase nK_pde/nT_pde or use theta=1.0"
            )

        c_on_strikes = np.empty((nT, strikes.size), dtype=float)
        for n in range(nT):
            c_on_strikes[n, :] = np.interp(strikes, K_grid, C[n, :], left=C[n, 0], right=C[n, -1])

        c_market = np.empty((tenors.size, strikes.size), dtype=float)
        for j in range(strikes.size):
            c_market[:, j] = np.interp(tenors, T_grid, c_on_strikes[:, j], left=c_on_strikes[0, j], right=c_on_strikes[-1, j])

        return c_market

    def calibrate(self, vol_matrices, strikes_list, tenors_list, vol_floor=1e-8,
                  max_iter=250, smooth_w=0.001, lr=0.07, tol=1e-8):
        vol_inputs = self._to_asset_list(vol_matrices, self.n_assets)
        tenor_inputs = self._to_asset_list(tenors_list, self.n_assets)
        strike_inputs = self._to_asset_list(strikes_list, self.n_assets)

        self.local_vol_matrices_ = []
        self.vol_interpolators_ = []

        for i in range(self.n_assets):
            tenors = np.asarray(tenor_inputs[i], dtype=float)
            strikes = np.asarray(strike_inputs[i], dtype=float)
            raw_matrix = np.asarray(vol_inputs[i], dtype=float)
            if raw_matrix.shape != (tenors.size, strikes.size):
                raise ValueError(
                    f"vol matrix shape {raw_matrix.shape} does not match grid ({tenors.size}, {strikes.size})"
                )

            t_grid, k_grid = np.meshgrid(tenors, strikes, indexing="ij")
            if np.nanmax(raw_matrix) <= 3.0:
                implied = np.maximum(raw_matrix, vol_floor)
                t_eff = np.maximum(t_grid, 1e-12)
                sqrt_t = np.sqrt(t_eff)
                zr = np.asarray(self.r_zero_interps_[i](t_eff), dtype=float)
                zq = np.asarray(self.q_zero_interps_[i](t_eff), dtype=float)
                df_r = np.exp(-zr * t_eff)
                df_q = np.exp(-zq * t_eff)
                log_m = np.log(np.maximum(float(self.S0_[i] if self.n_assets > 1 else self.S0_), 1e-12) / np.maximum(k_grid, 1e-12))
                market_prices, _ = self._black_scholes_call_price_and_vega(
                    float(self.S0_[i] if self.n_assets > 1 else self.S0_),
                    k_grid,
                    t_eff,
                    sqrt_t,
                    zr,
                    zq,
                    df_r,
                    df_q,
                    log_m,
                    implied
                )
                initial_sigma = implied
            else:
                market_prices = raw_matrix
                initial_sigma = np.full_like(raw_matrix, 0.2, dtype=float)

            calibrated_sigma = self._calibrate_local_vol_surface(
                market_prices=market_prices,
                s0=float(self.S0_[i] if self.n_assets > 1 else self.S0_),
                tenors=tenors,
                strikes=strikes,
                r_zero_interp=self.r_zero_interps_[i],
                q_zero_interp=self.q_zero_interps_[i],
                initial_sigma=initial_sigma,
                vol_floor=vol_floor,
                smooth_w=smooth_w,
                lr=lr,
                max_iter=max_iter,
                tol=tol
            )

            local_matrix = LocalVolatilityMatrix(
                tenors,
                strikes,
                calibrated_sigma,
                vol_floor=vol_floor
            )
            self.local_vol_matrices_.append(local_matrix)
            self.vol_interpolators_.append(local_matrix.get_interpolator())

        self.state_ = CalibrationState.CALIBRATED
        return self

    def calibrate_v2(self, vol_matrices, strikes_list, tenors_list, *,
                     S0_list=None, r_curve=None, q_curve=None,
                     vol_floor=1e-8,
                     degree=3, n_knots_T=6, n_knots_K=10,
                     alpha=1e-2, lm_lambda0=1e-1,
                     max_iter=25, tol=1e-6,
                     use_log_sigma=True,
                     nK_pde=None, nT_pde=None, pde_theta=1.0,
                     weights=None,
                     return_diagnostics=False):
        """
        Calibrates local volatility by minimizing implied-vol residuals:
          J(theta) = 0.5 * ||r_iv(theta)||^2 + 0.5 * alpha * ||L(theta-theta0)||^2
        where IV_model is obtained as:
          theta -> sigma(T,K;theta) -> Dupire PDE call prices -> BS implied vol inversion.
        """
        _ = (max_iter, tol)
        vol_inputs = self._to_asset_list(vol_matrices, self.n_assets)
        tenor_inputs = self._to_asset_list(tenors_list, self.n_assets)
        strike_inputs = self._to_asset_list(strikes_list, self.n_assets)

        if S0_list is None:
            s0_inputs = [float(self.S0_)] if self.n_assets == 1 else [float(x) for x in np.asarray(self.S0_, dtype=float)]
        else:
            s0_inputs = self._to_asset_list(S0_list, self.n_assets)

        r_inputs = self._to_asset_list(r_curve, self.n_assets) if r_curve is not None else [None] * self.n_assets
        q_inputs = self._to_asset_list(q_curve, self.n_assets) if q_curve is not None else [None] * self.n_assets
        if isinstance(weights, str):
            w_inputs = [weights] * self.n_assets
        else:
            w_inputs = self._to_asset_list(weights, self.n_assets) if weights is not None else [None] * self.n_assets

        self.local_vol_matrices_ = []
        self.vol_interpolators_ = []
        diagnostics = {"per_asset": []}

        for i in range(self.n_assets):
            tenors = np.asarray(tenor_inputs[i], dtype=float)
            strikes = np.asarray(strike_inputs[i], dtype=float)
            iv_mkt = np.asarray(vol_inputs[i], dtype=float)
            s0_i = float(s0_inputs[i])

            if tenors.ndim != 1 or strikes.ndim != 1:
                raise ValueError("tenors and strikes must be 1D")
            if np.any(tenors <= 0.0) or np.any(np.diff(tenors) <= 0.0):
                raise ValueError("tenors must be strictly positive and increasing")
            if np.any(strikes <= 0.0) or np.any(np.diff(strikes) <= 0.0):
                raise ValueError("strikes must be strictly positive and increasing")
            if iv_mkt.shape != (tenors.size, strikes.size):
                raise ValueError(f"vol matrix shape {iv_mkt.shape} does not match ({tenors.size}, {strikes.size})")
            if np.any(~np.isfinite(iv_mkt)):
                raise ValueError("vol matrix contains NaN/Inf")
            if np.any(iv_mkt < 0.0):
                raise ValueError("vol matrix must be non-negative")

            r_used = self.r_interps_[i] if (r_inputs[i] is None and i < len(self.r_interps_)) else (0.0 if r_inputs[i] is None else r_inputs[i])
            q_used = self.q_interps_[i] if (q_inputs[i] is None and i < len(self.q_interps_)) else (0.0 if q_inputs[i] is None else q_inputs[i])

            int_r = self._integrate_curve(r_used, tenors)
            int_q = self._integrate_curve(q_used, tenors)
            df_r_t = np.exp(-int_r)
            df_q_t = np.exp(-int_q)
            f_t = np.maximum(s0_i * df_q_t / np.maximum(df_r_t, 1e-16), 1e-16)
            market_call_prices = self._bs_call_from_implied(
                s0=s0_i, strikes=strikes, tenors=tenors, implied_vols=iv_mkt, int_r=int_r, int_q=int_q, vol_floor=vol_floor
            )

            t_max = float(np.max(tenors))
            k_min = float(np.min(strikes))
            k_max = float(np.max(strikes))
            k_span = max(k_max - k_min, 1e-8)
            k_pad = 0.05 * k_span
            knots_t = self._make_open_uniform_knots(0.0, t_max, int(degree), int(n_knots_T))
            knots_k = self._make_open_uniform_knots(max(1e-8, k_min - k_pad), k_max + k_pad, int(degree), int(n_knots_K))
            basis_t = self._bspline_design_matrix(tenors, knots_t, int(degree))
            basis_k = self._bspline_design_matrix(strikes, knots_k, int(degree))
            n_bt, n_bk = basis_t.shape[1], basis_k.shape[1]
            d2_t = self._second_diff_matrix(n_bt)
            d2_k = self._second_diff_matrix(n_bk)

            theta_vec, sigma0_init = self._initialize_theta_from_market(
                market_vols=np.maximum(iv_mkt, vol_floor),
                basis_t=basis_t,
                basis_k=basis_k,
                strikes=strikes,
                s0=s0_i,
                use_log_sigma=bool(use_log_sigma),
                vol_floor=vol_floor
            )
            p = theta_vec.size
            if p > 180:
                # TODO: replace FD by adjoint/AAD for scalability.
                pass

            w_mat = self._prepare_iv_weight_matrix(w_inputs[i], iv_mkt, strikes, tenors, f_t, df_r_t)
            sqrt_w = np.sqrt(np.maximum(w_mat, 0.0))

            eps_rel = 1e-3
            eps_abs = 1e-8
            tol_grad = 1e-6
            max_backtracks_per_iter = 3
            lambda_lm = float(lm_lambda0)
            n_pde_solves = 0
            lm_history = []

            residual_all, c_model, iv_model, r_price, r_reg, r_reg_t, r_reg_k, n_fail_iv = self._build_iv_residual(
                theta_vec=theta_vec, s0=s0_i, tenors=tenors, strikes=strikes, iv_mkt=iv_mkt, sqrt_w=sqrt_w,
                r_curve=r_used, q_curve=q_used, int_r=int_r, int_q=int_q, vol_floor=vol_floor, alpha=float(alpha),
                d2_t=d2_t, d2_k=d2_k, knots_t=knots_t, knots_k=knots_k, degree=int(degree),
                use_log_sigma=bool(use_log_sigma), nK_pde=nK_pde, nT_pde=nT_pde, pde_theta=float(pde_theta),
                return_parts=True
            )
            n_pde_solves += 1
            obj = 0.5 * float(np.dot(residual_all, residual_all))
            obj_initial = obj
            iv_model_initial = iv_model.copy()
            n_fail_initial = int(n_fail_iv)

            for it in range(int(max_iter)):
                j_total = np.zeros((residual_all.size, p), dtype=float)
                for k in range(p):
                    eps_k = eps_rel * max(1.0, abs(theta_vec[k])) + eps_abs
                    theta_pert = theta_vec.copy()
                    theta_pert[k] += eps_k
                    r_pert, _, _, _ = self._build_iv_residual(
                        theta_vec=theta_pert, s0=s0_i, tenors=tenors, strikes=strikes, iv_mkt=iv_mkt, sqrt_w=sqrt_w,
                        r_curve=r_used, q_curve=q_used, int_r=int_r, int_q=int_q, vol_floor=vol_floor, alpha=float(alpha),
                        d2_t=d2_t, d2_k=d2_k, knots_t=knots_t, knots_k=knots_k, degree=int(degree),
                        use_log_sigma=bool(use_log_sigma), nK_pde=nK_pde, nT_pde=nT_pde, pde_theta=float(pde_theta),
                        return_parts=False
                    )
                    n_pde_solves += 1
                    j_total[:, k] = (r_pert - residual_all) / eps_k

                grad = j_total.T @ residual_all
                h_gn = j_total.T @ j_total
                grad_norm = float(np.linalg.norm(grad))
                grad_inf = float(np.linalg.norm(grad, ord=np.inf))

                accepted = False
                rho = np.nan
                step_norm = 0.0
                delta_best = np.zeros_like(theta_vec)
                obj_prev = obj
                n_fail_new = int(n_fail_iv)

                for _bt in range(max_backtracks_per_iter):
                    h_lm = h_gn + lambda_lm * np.eye(p, dtype=float)
                    try:
                        delta = np.linalg.solve(h_lm, -grad)
                    except np.linalg.LinAlgError:
                        lambda_lm *= 10.0
                        continue

                    step_norm = float(np.linalg.norm(delta))
                    step_max = 10.0 * float(np.linalg.norm(theta_vec)) + 1.0
                    if step_norm > step_max:
                        delta *= step_max / step_norm
                        step_norm = float(np.linalg.norm(delta))

                    theta_new = theta_vec + delta
                    residual_new, c_model_new, iv_model_new, r_price_new, r_reg_new, r_reg_t_new, r_reg_k_new, n_fail_new = self._build_iv_residual(
                        theta_vec=theta_new, s0=s0_i, tenors=tenors, strikes=strikes, iv_mkt=iv_mkt, sqrt_w=sqrt_w,
                        r_curve=r_used, q_curve=q_used, int_r=int_r, int_q=int_q, vol_floor=vol_floor, alpha=float(alpha),
                        d2_t=d2_t, d2_k=d2_k, knots_t=knots_t, knots_k=knots_k, degree=int(degree),
                        use_log_sigma=bool(use_log_sigma), nK_pde=nK_pde, nT_pde=nT_pde, pde_theta=float(pde_theta),
                        return_parts=True
                    )
                    n_pde_solves += 1
                    obj_new = 0.5 * float(np.dot(residual_new, residual_new))

                    pred = -float(np.dot(grad, delta) + 0.5 * np.dot(delta, h_gn @ delta))
                    pred = max(pred, 1e-16)
                    ared = obj - obj_new
                    rho = ared / pred

                    if (obj_new < obj) or (rho > 1e-3):
                        theta_vec = theta_new
                        residual_all = residual_new
                        c_model = c_model_new
                        iv_model = iv_model_new
                        r_price = r_price_new
                        r_reg = r_reg_new
                        r_reg_t = r_reg_t_new
                        r_reg_k = r_reg_k_new
                        n_fail_iv = int(n_fail_new)
                        obj = obj_new
                        lambda_lm = max(1e-12, 0.3 * lambda_lm)
                        accepted = True
                        delta_best = delta
                        break

                    lambda_lm *= 10.0

                rel_obj = abs(obj_prev - obj) / max(obj_prev, 1.0)
                step_rel = (float(np.linalg.norm(delta_best)) / max(float(np.linalg.norm(theta_vec)), 1.0)) if accepted else np.inf
                lm_history.append({
                    "iter": it,
                    "obj": obj,
                    "lambda": lambda_lm,
                    "rho": rho,
                    "grad_norm": grad_norm,
                    "step_norm": step_norm,
                    "accepted": accepted,
                    "n_pde_solves": n_pde_solves,
                    "n_fail_iv_inversions": int(n_fail_iv),
                })

                if lambda_lm > 1e12:
                    raise RuntimeError("LM failed: ill-conditioned system (lambda_lm > 1e12)")
                if rel_obj < tol or grad_inf < tol_grad or step_rel < tol:
                    break

            local_sigma = self._sigma_from_theta_grid(
                theta_vec=theta_vec, basis_t=basis_t, basis_k=basis_k,
                use_log_sigma=bool(use_log_sigma), vol_floor=vol_floor
            )
            local_matrix = LocalVolatilityMatrix(tenors, strikes, local_sigma, vol_floor=vol_floor)
            self.local_vol_matrices_.append(local_matrix)
            self.vol_interpolators_.append(
                self._build_spline_sigma_interpolator(
                    theta_vec=theta_vec, knots_t=knots_t, knots_k=knots_k,
                    degree=int(degree), use_log_sigma=bool(use_log_sigma), vol_floor=vol_floor
                )
            )

            diagnostics["per_asset"].append({
                "asset_idx": i,
                "S0": s0_i,
                "integral_r": int_r,
                "integral_q": int_q,
                "market_iv": iv_mkt,
                "market_call_prices": market_call_prices,
                "model_call_prices_final": c_model,
                "iv_model_initial": iv_model_initial,
                "iv_model_final": iv_model,
                "weights": w_mat,
                "weights_mode": w_inputs[i] if isinstance(w_inputs[i], str) else "custom_or_uniform",
                "sigma0_init": sigma0_init,
                "objective_initial": obj_initial,
                "objective_final": obj,
                "r_price_norm_initial": float(np.linalg.norm((sqrt_w * (iv_model_initial - iv_mkt)).reshape(-1))),
                "r_price_norm_final": float(np.linalg.norm(r_price)),
                "r_reg_norm_final": float(np.linalg.norm(r_reg)),
                "lm_history": lm_history,
                "n_fail_iv_inversions_initial": n_fail_initial,
                "n_fail_iv_inversions_final": int(n_fail_iv),
                "n_pde_solves_total": n_pde_solves,
                "theta_shape": (n_bt, n_bk),
                "theta_final": theta_vec.copy(),
                "knots_t": knots_t,
                "knots_k": knots_k,
                "pde_nK": nK_pde,
                "pde_nT": nT_pde,
                "pde_theta": pde_theta,
            })

        self.state_ = CalibrationState.CALIBRATED
        if return_diagnostics:
            return self, diagnostics
        return self

####### Tree Simulator ########
class TreeSimulator:
    def __init__(self):
        pass

    @abstractmethod
    def simulate(self, dates, n_sims=10000):
        #Dates should be after init(i.e. >0)?
        #Dates are t times
        pass

    @abstractmethod
    def generateTree(self, dates):
        ### Meter suavización en último salto con BS si convergenceEnhacement
        pass

class BinomialTreeSimulator(TreeSimulator):
    def __init__(self, S0, r, q, sigma, deltat):
        self.S0_ = S0
        self.r_ = r
        self.q_ = q
        self.sigma_ = sigma
        self.deltat_ = deltat

        self.p_ = self.getProb()

        self.aux_fun_u = np.vectorize(lambda x: 1 if x>self.p_ else 0)
        self.aux_fun_d = np.vectorize(lambda x: 0 if x>self.p_ else 1)

        self.isJumpConstant_ = True

    def getDeltat(self):
        return self.deltat_

    @abstractmethod
    def getProb(self):
        pass

    @abstractmethod
    def getJump(self):
        pass
    
    def simulate(self, dates_integer, n_sims=10000):
        #TODO: dates should be a list of ascendent integers which we assume n-th step of the simulation,
        #time is i*deltat

        if self.isJumpConstant_:
            u,d = self.getJump(0)
            sims = np.random.random((n_sims, dates_integer[-1]))
            observations = self.S0_*np.cumprod(np.hstack((np.ones(shape=(n_sims,1)), self.aux_fun_u(sims)*u+self.aux_fun_d(sims)*d)), axis=1)
            
            ret = np.zeros((n_sims, len(dates_integer)))
            i = 0
            for d in dates_integer:
                ret[:, i] = observations[:, d]
                i+=1

            return ret

    #TODO: convergenceEnhacement=True en valoración sobre arbol aplica la formula de black-scholes para suavizar el último salto, pero podría aplicarse también en la generación del árbol para mejorar la convergencia de precios de opciones americanas, bermuda, etc. En ese caso habría que modificar el método generateTree para que el último salto se calcule con BS

    def generateTree(self, dates):
        N = dates[-1]+1
        tree = np.zeros((N, N))
        tree[0,0] = self.S0_

        if self.isJumpConstant_:
            u,d = self.getJump(0)

            for i in range(1, N):
                for j in range(i+1):
                    if j==0:
                        tree[j,i] = tree[j,i-1]*u
                    else:
                        tree[j,i] = tree[j-1,i-1]*d

        return tree
    
class JarrowRuddTree(BinomialTreeSimulator):
    def __init__(self, S0, r, q, sigma, deltat):
        super().__init__(S0, r, q, sigma, deltat)
        self.p_ = 0.5

        if isinstance(sigma, float):
            self.isJumpConstant_ = True
        elif isinstance(sigma, dict):
            self.isJumpConstant_ = False
        else:
            #TODO: lanzar error
            pass
        

    def getJump(self, date):
        deltat = self.deltat_
        if self.isJumpConstant_:
            u = np.exp((self.r_-self.q_)*deltat+self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
            d = np.exp((self.r_-self.q_)*deltat-self.sigma_*np.sqrt(deltat))*(2/(np.exp(self.sigma_*np.sqrt(deltat))+np.exp(-self.sigma_*np.sqrt(deltat))))
        else:
            pass
            
        return u,d

    def getProb(self):
        return 0.5 

##### Tendré que hacer wrappers con fechas concretas que usen esto por debajo
