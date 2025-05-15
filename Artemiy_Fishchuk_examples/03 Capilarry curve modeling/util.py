from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import least_squares, curve_fit, minimize
import sys
import math

class AbstractModel(ABC):
    def __init__(self, kv, pc):
        self.kv = np.array(kv)
        self.pc = np.array(pc)
        self._compute_params()

    def _compute_params(self):
        self.pc_vh = self.__compute_pcvh()
        self.kvo = min(self.kv)
        self.pc_kvo = self.pc[np.argmin(self.kv)]
    
    def __compute_pcvh(self):
        #Spaghetti code warning
        dy_dx = np.diff(self.pc)/np.diff(self.kv)
        d2y_dx2 = np.diff(dy_dx)/np.diff(self.kv[:-1])

        Pd = self.pc[np.argmin(np.abs(d2y_dx2))]
        return Pd

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    def predict(self, params, **kwargs):
        return self._compute(params, **kwargs)

class BruxKori(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)

    def _compute(self, n, mask):
        if mask.any() == None:
            mask = np.full(self.pc.shape, True)

        return self.kvo+(100-self.kvo)*(self.pc_vh/self.pc[mask])**(1/n)
    
    def _loss(self, n, mask):
        pred_kv = self._compute(n, mask)
        return self.kv[mask] - pred_kv
    
    def fit(self, n_start = 1):
        #train
        mask = (self.pc >= self.pc_vh)
        opt = least_squares(self._loss, n_start, args=[mask])
        #predict
        # mask = np.full(self.pc.shape, True)

        self.params = opt.x
        return self.params

class Kinetic(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)

    def _compute(self, pc, a, b):
        a,b = np.round((a, b))
        return ((pc-self.pc_kvo)/((self.pc_vh-pc + 1e-5)*a))**(1/b)+self.kvo
    
    def _loss(self, params):
        a,b = np.round(params, 5)
        return self.kv - self._compute(params)
    
    def fit(self, a=0.5, b=1.0):
        init_guess = [a,b]
        b0 = (0,1.0)
        b1 = (1.0,10.0)
        opt, _ = curve_fit(
            lambda Pc, a, b: self._compute(Pc, a, b),
            self.pc, self.kv,
            p0=init_guess,
            bounds=(b0, b1),
            max_nfev=5000
        )

        self.params = opt
        return self.params
    
    def predict(self, params):
        return self._compute(params)
        
    
    
class Optimal(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)
    
    def _compute(self, params):
        a,b = params
        alpha = (self.pc/a)**(1/b)
        return (alpha * self.kvo + 100)/(1+alpha)
    
    def _loss(self, params):
        return self.kv - self._compute(params)
    
    def fit(self, a=1, b=1):
        #train
        params = [a, b]
        opt = least_squares(self._loss, params, bounds=((0,0), (1,1)))
        self.params = opt.x

        return self.params

class Tomira(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)
    
    def _compute(self, G, mask):
        if mask.any() == None:
            mask = np.full(self.pc.shape, True)
        return self.kvo + (100-self.kvo)*(1- np.exp(G/np.log(self.pc_vh/(self.pc[mask] + 1e-6))))
    
    def _loss(self, G, mask):
        return self.kv[mask] - self._compute(G, mask)
    
    def fit(self, G=1):
        #train
        mask = (self.pc >= self.pc_vh)
        opt = least_squares(self._loss, G, args=[mask])
        self.params = opt.x
        
        return self.params
    
class Trigonometric(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)
    
    def _compute(self, Pc, A, B, C):
        """Тригонометрическая модель капиллярного давления."""
        angle = (np.pi/2 - np.arctan((Pc - A)/B))
        norm_sat = (angle/np.pi)**C
        return np.clip(norm_sat * (100 - self.kvo) + self.kvo, 0, 100)
    
    def _loss(self):
        pass
    
    def fit(self):
        """Подбор параметров и получение прогноза."""
        # Начинаем с начальных приближений
        init_guess = [np.median(self.pc), 1.0, 1.0]
        
        # Определяем ограничения на параметры
        lower_bounds = [0, 0.01, 0.1]
        upper_bounds = [10, 10, 10]
        
        # Оптимизация параметров
        params, _ = curve_fit(
            lambda Pc, A, B, C: self._compute(Pc, A, B, C),
            self.pc, self.kv,
            p0=init_guess,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=5000
        )
        
        
        return params
    





#depricated
