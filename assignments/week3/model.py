import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    A simple multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.CELU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            hidden_count: The number of layers L in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.activation = activation
        self.initializer = initializer

        self.layers = torch.nn.ModuleList()

        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(input_size, hidden_size)]
            self.initializer(self.layers[0].weight.data)
            input_size = hidden_size
        self.out = torch.nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = self.activation(layer(x))

        return self.out(x)
