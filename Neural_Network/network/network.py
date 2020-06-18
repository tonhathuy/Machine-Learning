
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None # loss fuc
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        """
        input: [[1,3]] -> 1, [[1,3],[3,5]...]
        return: ket qua du doan
        """
        result = []
        n = len(input)
        for i in range(n):
            output = input[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result


    def fit(self, x_train, y_train, learning_rate, epochs):
        # epochs: so lan train 1 table
        n = len(x_train)

        for i in range(epochs):
            err_total = 0 
            for j in range(n):
                # forward_propagation
                output = y_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                #tinh lỗi từng sample
                err_total += self.loss(y_train[j], output)

                #backward_propagation
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err_total = err_total / n

            print('epoch : %d/%d error total = %f'%(i, epochs, err_total))

