import numpy as np
import math
class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, seed=None):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        assert len(x)==len(y)
        self.shuffle = shuffle
        self.sample_size = len(x)

    def __iter__(self):
        perm_idx = np.arange(self.sample_size)
        if self.shuffle :
            perm_idx = self.rng.permutation(self.sample_size)

        for i in range(0,self.sample_size,self.batch_size) :
            idx = perm_idx[i :i+self.batch_size]
            yield self.x[idx],self.y[idx]

    def __len__(self):
        return math.ceil(self.sample_size/self.batch_size)
