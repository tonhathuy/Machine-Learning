from layer import Layer


class ActivationLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, activation_prime):
        """
            input_shape: dau vao input mang (1,4)
            output_shape ....
            activation C
            activation_prime : C'
        """
        self.input_shape = input_shape
        self.ouput_shape = output_shape
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        """
        input Z = (W*x) + bias
        """
        # save input, output khi cần dùng
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # E(h) = Eo x Wo x R'(Zh)
        # FC = Eo x Wo = output_error 
        # activation_prime = R'(Zh)
        return self.activation_prime(self.input)*output_error