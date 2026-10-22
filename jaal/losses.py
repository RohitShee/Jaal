import numpy as np
class Loss:
    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.forward(pred,target)

class MSELoss(Loss):
    def forward(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)
        self.pred = pred
        self.target = target
        mseloss = np.mean((pred-target)**2)
        return mseloss

    def backward(self):
        errors = self.pred-self.target
        return 2/self.pred.size*errors

class BCELoss(Loss):
    def forward(self, pred, target):
        pred = np.asarray(pred)
        target = np.asarray(target)
        self.pred = np.clip(pred,1e-12,1-1e-12)
        self.target = target
        bceloss = -np.mean(self.target*np.log(self.pred)+(1-self.target)*np.log(1-self.pred))
        return bceloss


    def backward(self):
        return 1/self.pred.size*(-self.target/self.pred+(1-self.target)/(1-self.pred))
