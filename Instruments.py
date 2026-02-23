from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from Utils import discount_from_zero_curve

class Instrument(ABC):
    def __init__(self):
        super().__init__()

class Option(Instrument):
    def __init__(self):
        super().__init__()
        self.models = dict()

class EquityOption(Option):
    def __init__(self):
        super().__init__()

    def price(self, model='BS', **kwargs):
        if model not in self.models:
            available = ", ".join(sorted(self.models.keys())) if self.models else "<none>"
            raise ValueError(f"Unknown model '{model}'. Available models: {available}")
        
        return self.models[model](**kwargs)
    
    def priceMC(self, simulator, payoff, dates, n_sims, dates_obs=None, discount_curve=None, **kwargs):
        if dates_obs is None:
            dates_obs = dates

        if discount_curve is None:
            discounts = [self.discount(date) for date in dates]
        else:
            discounts = [discount_from_zero_curve(discount_curve, date) for date in dates]
        sims = simulator.simulate(dates_obs, n_sims=n_sims)
        
        if len(dates) == 1:
            return np.mean( np.dot([payoff.cash_flows(sims[i,0], **kwargs) for i in range(n_sims)], discounts) )
        else:
            return np.mean( np.dot([payoff.cash_flows(sims[i,:], **kwargs) for i in range(n_sims)], discounts) )
        
    def priceTreeMC(self, simulator, payoff, dates, n_sims, **kwargs):
        deltat = simulator.getDeltat()
        dates_integer = [round(date/deltat) for date in dates]
        discounts = [self.discount(date) for date in dates]
        sims = simulator.simulate(dates_integer, n_sims=n_sims)
        
        if len(dates) == 1:
            return np.mean( np.dot([payoff.cash_flows(sims[i,0], **kwargs) for i in range(n_sims)], discounts) )
        else:
            return np.mean( np.dot([payoff.cash_flows(sims[i,:], **kwargs) for i in range(n_sims)], discounts) )

    
    def priceTree(self, simulator, dates, payoff, funLast=None):
        tree = simulator.generateTree(dates)
        p = simulator.getProb()
        deltat = simulator.getDeltat()

        N = tree.shape[0]

        disc = self.discount(deltat)

        value_tree = np.full((N,N), np.nan)
        for i in range(N):
            value_tree[i,N-1] = payoff.value( tree[i,N-1] )
        
        if funLast is not None:
            for i in range(N):
                val = funLast(tree[i, N-2])
                if payoff.hasDecision(deltat*(N-2), deltat):
                    value_tree[i, N-2] = max(payoff.value(tree[i,N-2]), val) 
                else:
                    value_tree[i, N-2] = val

            for j in range(N-3, -1, -1):
                for i in range(j+1):
                    val = (p*value_tree[i,j+1]+(1-p)*value_tree[i+1,j+1])*disc
                    if payoff.hasDecision(deltat*j, deltat):
                        value_tree[i,j] = max(payoff.value(tree[i,j]), val)
                    else:
                        value_tree[i,j] = val
        else:
            for j in range(N-2, -1, -1):
                for i in range(j+1):
                    val = (p*value_tree[i,j+1]+(1-p)*value_tree[i+1,j+1])*disc
                    if payoff.hasDecision(deltat*j, deltat):
                        value_tree[i,j] = max(payoff.value(tree[i,j]), val)
                    else:
                        value_tree[i,j] = val

        return value_tree[0,0]


    def getImpliedVol(self, price, model='BS', **kwargs):
        return opt.fsolve( lambda x: (self.price(model=model, sigma=x, **kwargs) - price)**2, 0.1)[0]
    
    def getBSImpliedVol(self, price, **kwargs):
        return self.getImpliedVol(price, model='BS', **kwargs)
    
    def getBachelierImpliedVol(self, price):
        return self.getImpliedVol(price, model='Bachelier')
    
    @abstractmethod
    def maturityDiscount(self):
        pass

    @abstractmethod
    def discount(self, date):
        pass
    


class OneStrikeOption(EquityOption):
    def __init__(self, S0, K, r, q, sigma, T):
        
        self.S0_ = S0
        self.K_ = K
        self.r_ = r
        self.q_ = q
        self.T_ = T
        self.sigma_flat_ = sigma

    def _validate_dupire_grid(self, tenors, strikes, vol_matrix):
        tenors = np.asarray(tenors, dtype=float)
        strikes = np.asarray(strikes, dtype=float)
        vol_matrix = np.asarray(vol_matrix, dtype=float)

        if tenors.ndim != 1 or strikes.ndim != 1:
            raise ValueError("tenors and strikes must be 1D arrays")
        if np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        if np.any(np.diff(strikes) <= 0):
            raise ValueError("strikes must be strictly increasing")
        if vol_matrix.shape != (tenors.size, strikes.size):
            raise ValueError(
                f"vol_matrix shape {vol_matrix.shape} does not match grid ({tenors.size}, {strikes.size})"
            )

    def _build_dupire_simulator(
        self,
        *,
        r_zero=None,
        q_zero=None,
        vol_matrix=None,
        strikes=None,
        tenors=None,
        vol_floor=1e-8
    ):
        from SimulatorEngine import DupireSimulator

        if vol_matrix is None or strikes is None or tenors is None:
            raise ValueError("vol_matrix, strikes, and tenors are required for Dupire calibration")

        self._validate_dupire_grid(tenors, strikes, vol_matrix)
        r_input = self.r_ if r_zero is None else r_zero
        q_input = self.q_ if q_zero is None else q_zero

        simulator = DupireSimulator(self.S0_, r_input, q_input, n_assets=1)
        simulator.calibrate(vol_matrix, strikes, tenors, vol_floor=vol_floor)
        return simulator

    def _resolve_dupire_simulator(
        self,
        simulator=None,
        *,
        r_zero=None,
        q_zero=None,
        vol_matrix=None,
        strikes=None,
        tenors=None,
        vol_floor=1e-8
    ):
        if simulator is not None:
            if getattr(simulator, "state_", None) != "CALIBRATED":
                raise ValueError("Provided Dupire simulator is not calibrated")
            return simulator

        return self._build_dupire_simulator(
            r_zero=r_zero,
            q_zero=q_zero,
            vol_matrix=vol_matrix,
            strikes=strikes,
            tenors=tenors,
            vol_floor=vol_floor
        )


