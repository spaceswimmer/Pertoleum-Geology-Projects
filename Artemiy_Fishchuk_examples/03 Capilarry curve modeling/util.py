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
    
    def __compute_pcvh(self):
        #Spaghetti code warning
        dy_dx = np.diff(self.pc)/np.diff(self.kv)
        d2y_dx2 = np.diff(dy_dx)/np.diff(self.kv[:-1])

        Pd = self.pc[np.argmin(np.abs(d2y_dx2))+2]
        return Pd

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def _loss(self):
        pass

class BruxKori(AbstractModel):
    def __init__(self, kv, pc):
        super().__init__(kv, pc)

    def _compute(self, n, mask ):
        # if not mask.all():
        #     mask = np.full(self.pc.shape, True)
        return self.kvo+(100-self.kvo)*(self.pc_vh/self.pc[mask])**(1/n)
    
    def _loss(self, n, mask):
        pred_kv = self._compute(n, mask)
        return self.kv[mask] - pred_kv
    
    def predict(self, n_start = 1):
        mask = (self.pc >= self.pc_vh)
        opt = least_squares(self._loss, n_start, args=[mask])
        mask = np.full(self.pc.shape, True)

        self.pred = self._compute(opt.x[0], mask)
        self.pred[self.pred>100] = 100
        self.pred[self.pred<0] = 0
        
        return self.pred

