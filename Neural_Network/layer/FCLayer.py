from layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        """
        input_shape: (1, 3)
        output_shape: (1, 4)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) -0.5

    @abstractclassmethod
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input*self.weights) + self.bias
        # output = (1x3) * (3x4) = (1x4)
        return self.output

    @abstractclassmethod
    def backward_propagation(self, output_error, learning_rate):
        # E(h) = Eo x Wo x R'(Zh)
        # FC = Eo x Wo
        # dweight = C'(Wo) = Eo*H = error lớp trc * input
        
        curent_layer_err = np.dot(output_error, self.weights.T)
        # output (1x4) weight (3x4)
        # chuyển vị weight ==> (1x4)*(4x3) = (1x3)
        dweight = np.dot(self.input, output_error)
        # vi input có dim=(1,3) ko nhân dc với output (1x4)
        # ==> chuyen vị input (3x1)*(1x4) = (3x4)

        # update weight, bias
        self.weights -= dweight*learning_rate
        self.bias -= learning_rate*output_error

        return curent_layer_err #de su dung cho lan truyen tiep theo
