from Instruments import EquityOption, OneStrikeOption
from SimulatorEngine import BlackScholesSimulator, Black76Simulator, JarrowRuddTree
from Payoffs import EuropeanCallPayOff, EuropeanPutPayOff, AsianCallPayoff, AsianPutPayoff, DigitalCallPayoff, DigitalPutPayoff, BarrierCallPayoff, BarrierPutPayoff, AmericanCallPayOff, AmericanPutPayOff, BermudanCallPayOff, BermudanPutPayOff, AutocallMultiCouponPayoff
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

####### Europeas ########
class EuropeanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = EuropeanCallPayOff(self.K_)
        elif c_p == 'p':
            self.payoff_ = EuropeanPutPayOff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS, 'JarrowRuddTree': self.priceJarrowRuddTree}

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
            d1 = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
            d2 = d1 - sigma * np.sqrt(self.T_)
        else:
            sigma = self.sigma_flat_
            d1 = self.d1_
            d2 = self.d2_

        if self.c_p == 'p':
            w = -1
        else:
            w = 1

        if simulation:
            return self.priceBSMC(sigma, n_sims=n_sims)
        else:
            return w*self.S0_ * np.exp(-self.q_ * self.T_) * st.norm.cdf(w*d1) - w*self.K_ * np.exp(-self.r_ * self.T_) * st.norm.cdf(w*d2)


    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)  
    
    def priceJarrowRuddTree(self, sigma=None, simulation=False, deltat=1/252, n_sims=10000):
        if simulation:
            return self.priceJarrowRuddTreeMC(sigma=sigma, deltat=deltat, n_sims=n_sims)
        else:
            simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
            return self.priceTree(simulator,[round(self.T_/deltat)], self.payoff_)
        
    def priceJarrowRuddTreeMC(self, sigma=None, deltat=1/252, n_sims=10000):
        if sigma == None:
            sigma = self.sigma_flat_
        
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)

        return self.priceTreeMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)

    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)

class EuropeanCall(EuropeanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='c')

class EuropeanPutOption(EuropeanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='p')

### TODO: price con superficie de vol: en tiempo solo o en tiempo y nivel subyacente 
    ### -> definición de simuladores y calibración

##Hacer un enum o así con las griegas que se pueden devolver 

######## Asiáticas ########
class AsianOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, dates, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.dates_ = dates
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = AsianCallPayoff(self.K_)
        elif c_p == 'p':
            self.payoff_ = AsianPutPayoff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS}

    def priceBS(self, sigma=None, n_sims=10000):
        #TODO: calcular con integración numérica de número de observaciones  ?
        if sigma == None:
            sigma = self.sigma_flat_
        return self.priceBSMC(sigma, n_sims=n_sims)

    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, self.dates_, n_sims=n_sims)
    
    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)
    
    ### TODO: price con superficie de vol: en tiempo solo o en tiempo y nivel subyacente 
    ### -> definición de simuladores y calibración
    
class AsianCallOption(AsianOption):
    def __init__(self, S0, K, r, q, sigma, dates, T):
        super().__init__(S0, K, r, q, sigma, dates, T, c_p='c')

class AsianPutOption(AsianOption):
    def __init__(self, S0, K, r, q, sigma, dates, T):
        super().__init__(S0, K, r, q, sigma, dates, T, c_p='p')

######## Digitales ########

class DigitalOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = DigitalCallPayoff(self.K_)
        elif c_p == 'p':
            self.payoff_ = DigitalPutPayoff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS}

    def getBSImpliedVol(self, price):
        return opt.fsolve( lambda x: (self.priceBS(sigma=x) - price)**2, 0.1)

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma == None:
            sigma = self.sigma_flat_
            d1 = self.d1_
            d2 = self.d2_
        else:
            d1 = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
            d2 = d1 - sigma * np.sqrt(self.T_)

        if simulation:
            return self.priceBSMC(sigma=sigma, n_sims=n_sims)
        else:
            return st.norm.cdf(d2) if self.c_p == 'c' else 1-st.norm.cdf(d2)

    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)
    
    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)

class DigitalCallOption(DigitalOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, 'c')

class DigitalPutOption(DigitalOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, 'p')

###### BarrierOption ######

class BarrierOption(OneStrikeOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, c_p, u_d, activation=True):
        super().__init__(S0, K, r, q, sigma, T)
        self.d1_ = (np.log(self.S0_ / self.K_) + (self.r_ - self.q_ + 0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = BlackScholesSimulator(self.S0_, self.r_, self.q_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = BarrierCallPayoff(self.K_, barrier, u_d=u_d, activation=activation)
        elif c_p == 'p':
            self.payoff_ = BarrierPutPayoff(self.K_, barrier, u_d=u_d, activation=activation)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS}

    def priceBS(self, dates, sigma=None, n_sims=10000):
        return self.priceBSMC(self, dates, sigma=sigma, n_sims=n_sims)
    
    def priceBSMC(self, dates, sigma=None, n_sims=10000):
        ### Dates should be a list and contain maturity at last element
        if abs(dates[-1] - self.T_) > 0.001:
            #TODO: lanzar error
            pass

        if sigma != None:
            simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)
        else:
            simulator = self.simulatorBS_
        return self.priceMC(simulator, self.payoff_, dates, n_sims=n_sims)
    
    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)
    
class BarrierCallOption(BarrierOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, u_d, activation=True):
        super().__init__(S0, K, barrier, r, q, sigma, T, c_p='c', u_d=u_d, activation=activation)
        

class BarrierPutOption(BarrierOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T, u_d, activation=True):
        super().__init__(S0, K, barrier, r, q, sigma, T, c_p='p', u_d=u_d, activation=activation)

##### Clases ayuda ######

##### Call #####
class DownOutCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=False)

class DownInCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=True)
    
class UpOutCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=False)

class UpInCallPayoff(BarrierCallOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=True)

##### Put #####
class DownOutPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=False)

class DownInPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='d', activation=True)
    
class UpOutPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=False)

class UpInPutPayoff(BarrierPutOption):
    def __init__(self, S0, K, barrier, r, q, sigma, T):
        super().__init__(S0, K, barrier, r, q, sigma, T, u_d='u', activation=True)

##### Opciones vanilla sobre futuros ####
class FutureEuropeanVanillaOption(OneStrikeOption):
    def __init__(self, F0, K, r, sigma, T, c_p):
        super().__init__(F0, K, r, 0, sigma, T)
        self.F0_ = F0
        self.d1_ = (np.log(self.F0_ / self.K_) + (0.5 * self.sigma_flat_**2) * self.T_) / (self.sigma_flat_ * np.sqrt(self.T_))
        self.d2_ = self.d1_ - self.sigma_flat_ * np.sqrt(self.T_)
        self.simulatorBS_ = Black76Simulator(self.F0_, self.sigma_flat_)
        if c_p == 'c':
            self.payoff_ = EuropeanCallPayOff(self.K_)
        elif c_p == 'p':
            self.payoff_ = EuropeanPutPayOff(self.K_)
        else: 
            pass #TODO: lanzar error

        self.c_p = c_p
        self.models = {'BS': self.priceBS}

    def priceBS(self, sigma=None, simulation=False, n_sims=10000):
        if sigma != None:
            d1 = (np.log(self.F0_ / self.K_) + (0.5 * sigma**2) * self.T_) / (sigma * np.sqrt(self.T_))
            d2 = d1 - sigma * np.sqrt(self.T_)
        else:
            sigma = self.sigma_flat_
            d1 = self.d1_
            d2 = self.d2_

        if self.c_p == 'p':
            w = -1
        else:
            w = 1

        if simulation:
            return self.priceBSMC(sigma, n_sims=n_sims)
        else:
            return  w*self.maturityDiscount()*(self.F0_ * st.norm.cdf(w*d1) - self.K_ * st.norm.cdf(w*d2))


    def priceBSMC(self, sigma=None, n_sims=10000):
        if sigma != None:
            simulator = Black76Simulator(self.F0_, sigma)
        else:
            simulator = self.simulatorBS_

        return self.priceMC(simulator, self.payoff_, [self.T_], n_sims=n_sims)  
    
    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)
    
class FutureEuropeanCallOption(FutureEuropeanVanillaOption):
    def __init__(self, S0, K, r, sigma, T):
        super().__init__(S0, K, r, sigma, T, c_p='c')

class FutureEuropeanPutOption(FutureEuropeanVanillaOption):
    def __init__(self, S0, K, r, sigma, T):
        super().__init__(S0, K, r, sigma, T, c_p='p')

######## Americanas #######

class AmericanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        if c_p=='c':
            self.payoff_ = AmericanCallPayOff(self.K_)
        elif c_p=='p':
            self.payoff_ = AmericanPutPayOff(self.K_)
        else:
            #TODO: lanzar error
            pass

        self.c_p = c_p
        self.models = {'JarrowRuddTree': self.priceJarrowRuddTree}
    
    def price(self, model='JarrowRuddTree', **kwargs):
        return super().price(model, **kwargs)
    
    def priceJarrowRuddTree(self, sigma=None, deltat=1/252):
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
        return self.priceTree(simulator, [round(self.T_/deltat)], self.payoff_)

    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)
    
class AmericanCallOption(AmericanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='c')

class AmericanPutOption(AmericanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T):
        super().__init__(S0, K, r, q, sigma, T, c_p='p')

######## Bermuda ##########
class BermudanVanillaOption(OneStrikeOption):
    def __init__(self, S0, K, r, q, sigma, T, dates, c_p):
        super().__init__(S0, K, r, q, sigma, T)
        self.dates_ = dates
        if c_p=='c':
            self.payoff_ = BermudanCallPayOff(self.K_, dates)
        elif c_p=='p':
            self.payoff_ = BermudanPutPayOff(self.K_, dates)
        else:
            #TODO: lanzar error
            pass

        self.c_p = c_p
        self.models = {'JarrowRuddTree': self.priceJarrowRuddTree}
    
    def price(self, model='JarrowRuddTree', **kwargs):
        return super().price(model, **kwargs)
    
    def priceJarrowRuddTree(self, sigma=None, deltat=1/252):
        simulator = JarrowRuddTree(self.S0_, self.r_, self.q_, sigma, deltat=deltat)
        return self.priceTree(simulator, [round(self.T_/deltat)], self.payoff_)

    def maturityDiscount(self):
        return np.exp(-self.r_*self.T_)
    
    def discount(self,date):
        return np.exp(-self.r_*date)
    
class BermudanCallOption(BermudanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T, dates):
        super().__init__(S0, K, r, q, sigma, T, dates, c_p='c')

class BermudanPutOption(BermudanVanillaOption):
    def __init__(self, S0, K, r, q, sigma, T, dates):
        super().__init__(S0, K, r, q, sigma, T, dates, c_p='p')

######## Estructurados ####

class AutocallOption(EquityOption):
    def __init__(self, notional, S0, r, q, dates, dates_obs, coupons, coup_barriers, autocall_barrier, maturity_options, isAcumulative, participations, autocall_start_period=1):
        self.payoff_ = AutocallMultiCouponPayoff(notional, coupons, coup_barriers, autocall_barrier, isAcumulative, maturity_options, participations, autocall_start_period=autocall_start_period)
        self.dates_ =dates
        self.dates_obs_ =dates_obs
        self.S0_ = S0
        self.r_ = r
        self.q_ = q
        self.T_ = self.dates_[-1]

        self.models = {'BS':self.priceBS}

    def priceBS(self, sigma, n_sims=10000):
        simulator = BlackScholesSimulator(self.S0_, self.r_, self.q_, sigma)

        return self.priceMC(simulator, self.payoff_, self.dates_, n_sims, dates_obs=self.dates_obs_)
    
    def maturityDiscount(self):
        return self.discount(self.T_)
    
    def discount(self, date):
        return np.exp(-self.r_*date)


class AutocallPutOption(AutocallOption):
    def __init__(self, notional, S0, r, q, dates, dates_obs, coupon, coup_barrier, autocall_barrier, put_strike, isAcumulative, isLeveraged, autocall_start_period=1):
        part = 1/put_strike if isLeveraged else 1
        super().__init__(notional, S0, r, q, dates, dates_obs, [coupon], [coup_barrier], autocall_barrier, [EuropeanPutPayOff(put_strike)], isAcumulative, [part], autocall_start_period=autocall_start_period)

if __name__=='__main__':
    dates = [0.25*i for i in range(1,37)]
    dates_obs = [0.25*i - 3/252 for i in range(1,37)]
    apop = AutocallPutOption(100, 1, 0.021, 0.0385, dates, dates, 0.025, 0.65, 1, 0.65, True, True)
    apop2 = AutocallPutOption(100, 1, 0.021, 0.0485, dates, dates, 0.025, 0.65, 1, 0.65, True, True)
    print(apop.price(sigma=0.27, n_sims=100000))

    n_sims = 100000
    spot = 100
    strike = 100
    r = 0.05
    q = 0.01
    sigma = 0.2
    T = 1

    ##### Test europeas
    op = EuropeanCallOption(spot, strike , r, q, sigma, T)
    op2 = EuropeanPutOption(spot, strike , r, q, sigma, T)

    price = op.price(model='BS', sigma=0.2)

    print('Opciones europeas:')
    print(op.priceBS(0.2))
    print(op.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op2.priceBS(0.2))
    print(op2.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(op.getImpliedVol(price))


    ##### Test asiáticas
    dates = [1]
    aop = AsianCallOption(spot, strike , r, q, sigma, dates, T)
    aop2 = AsianPutOption(spot, strike , r, q, sigma, dates, T)

    print('Opciones asiáticas con solo fecha 1:')
    print(aop.priceBS(n_sims=n_sims))
    print(aop2.priceBS(n_sims=n_sims))

    dates = [0.25,0.5,0.75,1]
    aop = AsianCallOption(spot, strike , r, q, sigma, dates, T)
    aop2 = AsianPutOption(spot, strike , r, q, sigma, dates, T)
    print('Opciones asiáticas con fechas 0.25,0.5,0.75,1:')
    print(aop.priceBS(n_sims=n_sims))
    print(aop2.priceBS(n_sims=n_sims))


    ##### Test digitales
    strike = 120
    dop = DigitalCallOption(spot, strike , r, q, sigma, T)
    dop2 = DigitalPutOption(spot, strike , r, q, sigma, T)

    print('Opciones digitales:')
    print(dop.priceBS(0.2))
    print(dop.priceBS(0.2, simulation=True, n_sims=n_sims))
    print(dop2.priceBS(0.2))
    print(dop2.priceBS(0.2, simulation=True, n_sims=n_sims))

    ##### Test opciones sobre futuros
    strike = 120
    fop = FutureEuropeanCallOption(spot, strike, r, sigma, T)
    fop2 = FutureEuropeanPutOption(spot, strike, r, sigma, T)

    print("Opciones sobre futuros:")
    print(fop.priceBS(0.2))
    print(fop.priceBS(0.2, simulation=True, n_sims=n_sims*3))
    print(fop2.priceBS(0.2))
    print(fop2.priceBS(0.2, simulation=True, n_sims=n_sims*3))

    ##### Test barrera
    strike = 100
    barrier = 120

    bop = UpInCallPayoff(spot, strike, barrier, r, q, sigma, T)
    aux11 = DigitalCallOption(spot, barrier, r, q, sigma, T)
    aux12 = EuropeanCallOption(spot, barrier, r,q,sigma,T)

    barrier2 = 80
    bop2 = DownInPutPayoff(spot, strike, barrier2, r, q, sigma, T)
    aux21 = DigitalPutOption(spot, barrier2, r, q, sigma, T)
    aux22 = EuropeanPutOption(spot, barrier2, r,q,sigma,T)

    print("Opciones con barrera: ")
    print("Call up&in a vencimiento es como digital+call en la barrera:")
    print(bop.priceBSMC([T], n_sims=n_sims*6))
    print( (barrier-strike)*aux11.priceBS( )+aux12.priceBS())

    print("Put down&in a vencimiento es como digital+call en la barrera:")
    print(bop2.priceBSMC([T], n_sims=n_sims*6))
    print((strike-barrier2)*aux21.priceBS( )+aux22.priceBS())

    #Test árbol
    print('JarrowRudd simulator:')
    print(JarrowRuddTree(spot, r, q, sigma, deltat=1/12).simulate(n_sims=1, dates_integer=[5]))
    
    print('Valoración simulación árbol:')
    print('Black-scholes teórico: ', op.priceBS(0.2))
    print('Valoración simulando binomial: ', op.price(model='JarrowRuddTree', sigma=0.2, simulation=True, n_sims=10000))

    print('Valoración en árbol:')
    print(op.price(model='JarrowRuddTree', sigma=0.2, n_sims=n_sims))

    #Test americanas: comprobar con un excel
    print('Valoración call americana:')
    op = EuropeanCallOption(spot, strike , r, 0, sigma, T)
    op2 = EuropeanPutOption(spot, strike , r, 0, sigma, T)

    aop = AmericanCallOption(spot, strike, r, 0, sigma, T)

    print('Europea: ',op.priceBS(0.2))

    print('Americana: ',aop.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Valoración put americana:')
    aop2 = AmericanPutOption(spot, strike, r, 0, sigma, T)
    
    price =aop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2)
    print('Europea: ', op2.priceBS(0.2))
    print('Americana: ', aop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Vola implícita: ', aop2.getImpliedVol(price, model='JarrowRuddTree', deltat=1/12))

    #Test bermuda
    bermop = BermudanCallOption(spot, strike, r, 0, sigma, T, [0.5])
    bermop2 = BermudanPutOption(spot, strike, r, 0, sigma, T, [0.5])
    
    print('Bermudas con fecha de ejercicio intermedia en 0.5: ')
    print('Call: ')
    print('Europea: ',op.priceBS(0.2))
    print('Bermuda: ',bermop.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Put: ')
    print('Europea: ',op2.priceBS(0.2))
    print('Bermuda: ',bermop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))

    print('Bermudas con fechas de ejercicio 0.25, 0.5, 0.75: ')
    bermop = BermudanCallOption(spot, strike, r, 0, sigma, T, [0.25, 0.5, 0.75])
    bermop2 = BermudanPutOption(spot, strike, r, 0, sigma, T, [0.25, 0.5, 0.75])

    print('Call: ',bermop.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))
    print('Put: ',bermop2.price(model='JarrowRuddTree', deltat=1/12, sigma=0.2))