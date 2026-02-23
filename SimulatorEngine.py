from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
from scipy.special import ndtr
from scipy.interpolate import interp1d
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

    def _calibrate_local_vol_surface(self, market_prices, s0, tenors, strikes, r_zero_interp, q_zero_interp, initial_sigma, vol_floor):
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
        smooth_w = 0.02
        lr = 0.15
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        max_iter = 80
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
            if np.isfinite(prev_loss) and abs(prev_loss - loss) <= 1e-10:
                break
            prev_loss = loss

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** it)
            v_hat = v / (1.0 - beta2 ** it)

            sigma = sigma - lr * m_hat / (np.sqrt(v_hat) + eps)
            sigma = np.maximum(sigma, vol_floor)

        return sigma

    def calibrate(self, vol_matrices, strikes_list, tenors_list, vol_floor=1e-8):
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
                vol_floor=vol_floor
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
