from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import least_squares
import sys

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
    def predict(self):
        pass

class BruxKori(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)

    def _compute(self, n, mask):
        return self.kvo+(100-self.kvo)*(self.pc_vh/self.pc[mask])**(1/n)
    
    def _loss(self, n, mask):
        pred_kv = self._compute(n, mask)
        return self.kv[mask] - pred_kv
    
    def predict(self, n_start = 1):
        mask = (self.pc >= self.pc_vh)
        opt = least_squares(self._loss, n_start, args=[mask])
        mask = np.full(self.pc.shape, True)

        self.pred = self._compute(opt.x, mask)
        self.pred[self.pred>100] = 100
        self.pred[self.pred<0] = 0
        
        return self.pred
    
class Kinetic(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)

    def _compute(self, params, mask):
        a,b = np.round(params, decimals=5)
        return ((self.pc[mask]-self.pc_kvo)/((self.pc_vh-self.pc[mask]+1e-5)*a))**(1/b)+self.kvo
    
    def _loss(self, params, mask):
        a,b = np.round(params, decimals=5)
        result = self.kv[mask] - self._compute(params, mask)
        return result
    
    def predict(self, a=1, b=1):
        #train
        params = [a, b]
        mask = (self.pc >= self.pc_vh)
        opt = least_squares(self._loss, params, args=[mask], bounds=((0,1), (1,10)))
        #predict
        mask = (self.pc > self.pc_vh)
        self.pred = self._compute(opt.x, mask)
        self.pred[self.pred>100] = 100
        self.pred[self.pred<0] = 0
        
        return self.pred
    
class Optimal(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)
    
    def _compute(self, params):
        a,b = params
        alpha = (self.pc/a)**(1/b)
        return (alpha * self.kvo + 100)/(1+alpha)
    
    def _loss(self, params):
        return self.kv - self._compute(params)
    
    def predict(self, a=1, b=1):
        #train
        params = [a, b]
        opt = least_squares(self._loss, params, bounds=((0,0), (1,1)))
        #predict
        self.pred = self._compute(opt.x)
        self.pred[self.pred>100] = 100
        self.pred[self.pred<0] = 0
        
        return self.pred
        