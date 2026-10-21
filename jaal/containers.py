from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers :
            x = layer(x)
        return x

    def backward(self, grad_output):
        for layer in self.layers[::-1] :
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers :
            params.extend(layer.parameters())
        return params

    def train(self):
        super().train()
        for layer in self.layers :
            layer.train()
        return self

    def eval(self):
        super().eval()
        for layer in self.layers :
            layer.eval()
        return self

