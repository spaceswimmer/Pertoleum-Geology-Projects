from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import least_squares, curve_fit, minimize
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
    def fit(self):
        pass

    def predict(self, params, mask = None):
        return self._compute(params, mask)

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

    def _compute(self, params):
        a,b = np.round(params, 5)
        return ((self.pc-self.pc_kvo)/((self.pc_vh-self.pc+1e-5)*a))**(1/b)+self.kvo
    
    def _loss(self, params):
        a,b = np.round(params, 5)
        return self.kv - self._compute(params)
    
    def fit(self, a=0.5, b=1.0):
        params = [a,b]
        b0 = (0,1.0)
        b1 = (1.0,10.0)
        opt = least_squares(self._loss, params, bounds = (b0, b1))

        self.params = opt.x
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
    
    def _compute(self, params):
        A, B, C = params
        return (1/2-np.arctan((self.pc-A)/B)/np.pi)**(1/C)
    
    def _loss(self, params):
        return self.kv - self._compute(params)
    
    def fit(self, A=2, B=2, C=2):
        #train
        params = [A, B, C]
        opt = least_squares(self._loss, params, bounds=([0, 0.01, 0.1], [10, 10, 10]))
        self.params = opt.x
        
        return self.params
    





#depricated
def plot_all_models(data, pc, model_name):
    """
    Function to iterate through all models, predict, and plot real and predicted curves.
    """
    num_models = len(data)
    num_cols = 4
    num_rows = math.ceil(num_models / num_cols)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    for idx, num in enumerate(data.keys()):
        mask = np.full(pc.shape, True)
        match model_name:
            case "BruxKori":   
                model = BruxKori(data[num]["kv"], pc)
            case "Kinetic":
                model = Kinetic(data[num]["kv"], pc)
                mask = (model.pc > model.pc_vh)
            case "Optimal":
                model = Optimal(data[num]["kv"], pc)
            case "Tomira":
                model = Tomira(data[num]["kv"], pc)
                mask = (model.pc >= model.pc_vh)
            case "Trigonometric":
                model = Trigonometric(data[num]["kv"], pc)
        
        predicted = model.predict()
        real = model.kv
        
        ax = axes[idx]
        ax.plot(real, pc, marker="o", linestyle="-", label="Real")
        ax.plot(predicted, pc[mask], marker="o", linestyle="--", label="Predicted")
        
        ax.set_xlabel("Kv, %")
        ax.set_ylabel("Pc, МПа")
        ax.set_yscale("log")
        ax.set_title(f"Model {num}")
        ax.hlines(y=model.pc_vh, xmin=30, xmax=100, label="pc_vh", colors="r")
        ax.legend()
    
    # Hide unused subplots
    for ax in axes[num_models:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()