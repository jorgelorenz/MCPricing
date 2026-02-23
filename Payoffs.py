from abc import ABC, abstractmethod
import numpy as np


class Payoff(ABC):
    @abstractmethod
    def value(self, observations, **kwargs):
        pass

    def hasDecision(self, date, deltat):
        return False
    
    def cash_flows(self, observations):
        if isinstance(observations, np.float64):
            res = np.zeros(1)
        else:
            res = np.zeros(len(observations))
        res[-1] = self.value(observations)
        return res

class EuropeanVanillaPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

class EuropeanCallPayOff(EuropeanVanillaPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class EuropeanPutPayOff(EuropeanVanillaPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

class AsianPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

class AsianCallPayoff(AsianPayoff):
    def value(self, observations):
        return max(0, np.mean(observations)-self.K_)

class AsianPutPayoff(AsianPayoff):
    def value(self, observations):
        return max(0, self.K_-np.mean(observations))

class DigitalPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K
    
class DigitalCallPayoff(DigitalPayoff):
    def value(self, observation):
        return 1 if observation > self.K_ else 0

class DigitalPutPayoff(DigitalPayoff):
    def value(self, observation):
        return 1 if observation < self.K_ else 0
    

class BarrierPayoff(Payoff):
    def __init__(self, K, barrier, u_d='u', activation=True):
        self.K_ = K
        self.barrier_ = barrier
        self.u_d_ = u_d
        self.activation_ = activation

    def isActiveAtMaturity(self, observations):
        mini = np.min(observations)
        maxi = np.max(observations)
        if self.activation_:
            if self.u_d_ == 'u':
                return self.barrier_ < maxi
            else:
                return self.barrier_ > mini
        else:
            if self.u_d_ == 'u':
                return self.barrier_ > maxi
            else:
                return self.barrier_ < mini
    
class BarrierCallPayoff(BarrierPayoff):
    def value(self, observations):
        if isinstance(observations, np.float64):
            last = observations
        else:
            last = observations[-1]

        if self.isActiveAtMaturity(observations):
            return max(last - self.K_, 0)
        else:
            return 0

class BarrierPutPayoff(BarrierPayoff):
    def value(self, observations):
        if isinstance(observations, np.float64):
            last = observations
        else:
            last = observations[-1]
            
        if self.isActiveAtMaturity(observations):
            return max(self.K_ - last, 0)
        else:
            return 0


class DownOutCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=False)

class DownInCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=True)
    
class UpOutCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=False)

class UpInCallPayoff(BarrierCallPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=True)

class DownOutPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=False)

class DownInPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='d', activation=True)
    
class UpOutPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=False)

class UpInPutPayoff(BarrierPutPayoff):
    def __init__(self, K, barrier):
        super().__init__(K, barrier, u_d='u', activation=True)

class AmericanPayoff(Payoff):
    def __init__(self, K):
        self.K_ = K

    def hasDecision(self, date, deltat):
        return True
    
class AmericanCallPayOff(AmericanPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class AmericanPutPayOff(AmericanPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

class BermudanPayoff(Payoff):
    def __init__(self, K, dates):
            self.K_ = K
            self.dates_ = np.array(dates)

    def hasDecision(self, date, deltat):
        return  (np.absolute(self.dates_ - date) < deltat/2).any()
    
class BermudanCallPayOff(BermudanPayoff):
    def value(self, observation):
        return max(0, observation-self.K_)
    
class BermudanPutPayOff(BermudanPayoff):
    def value(self, observation):
        return max(0, self.K_ - observation)

        
class AutocallMultiCouponPayoff(Payoff):
    def __init__(self, notional, coupons, coup_barriers, autocall_barrier, isAcumulative,maturity_payoffs, maturity_participations, autocall_start_period=1):
        self.notional_ = notional
        self.coupons_ = coupons
        self.cup_barriers_ = coup_barriers

        self.autocall_barrier_ = autocall_barrier
        self.isAcumulative_ = isAcumulative

        self.maturity_payoffs_ = maturity_payoffs
        self.maturity_participations_ = maturity_participations

        self.autocall_start_period = autocall_start_period-1

    def value(self, observations, discounts=None):
        return np.dot(self.cash_flows(observations), discounts)
    
    def cash_flows(self, observations):
        cf = np.zeros(len(observations))
        coup_acum = np.zeros(len(self.coupons_))
        for i, obs in enumerate(observations):
            for j, (c, b) in enumerate(zip(self.coupons_, self.cup_barriers_)):
                if obs > b:
                    cf[i] += (coup_acum[j] + c) * self.notional_
                    coup_acum[j] = 0
                elif self.isAcumulative_:
                    coup_acum[j] += c
                
            if obs > self.autocall_barrier_ and i >= self.autocall_start_period:
                cf[i] += self.notional_
                return cf
            
        cf[-1] += self.notional_     
        
        if len(self.maturity_payoffs_) > 0:
            cf[-1] -= self.notional_*np.dot([payoff.value(observations[-1]) for payoff in self.maturity_payoffs_], self.maturity_participations_)
            
        return  cf

class AutocallPayoff(AutocallMultiCouponPayoff):
    def __init__(self, notional, coupon, coup_barrier, autocall_barrier, isAcumulative,maturity_payoffs, maturity_participations, autocall_start_period=1):
        super().__init__(notional, [coupon], [coup_barrier], autocall_barrier, isAcumulative,maturity_payoffs, maturity_participations, autocall_start_period=autocall_start_period)


