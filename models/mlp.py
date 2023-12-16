import math
import torch
import torch.nn as nn
from torch.nn import MSELoss

class MLP(nn.Module):
    """MLP for working with transformer's output, numerical & categorical features
       in order to predict the price of a renting (regression task).
    """

    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dimensions, dropout_prob=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dimensions
        self.layer_neurons = [self.input_dim] + self.hidden_dim + [self.out_dim]
        assert self.hidden_layers == len(self.hidden_dim)
        self.activation_name = 'relu'
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList(list(map(
            self.xavier_weight_init,
            [nn.Linear(
                self.layer_neurons[i],
                self.layer_neurons[i + 1]
            ) for i in range(len(self.layer_neurons) - 2)
            ]
        )))
        self.dropout = nn.Dropout(dropout_prob)
        
        # final layer is created separately to allow for different activation function
        final_layer = nn.Linear(self.layer_neurons[-2], self.layer_neurons[-1])
        self.xavier_weight_init(final_layer, activation='linear')
        self.layers.append(final_layer)
        
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in self.layer_neurons[1:-1] # exclude input and output layers
        ])

    def xavier_weight_init(self, model, activation=None):
        if activation is None:
            activation = self.activation_name
        torch.nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain(activation))
        return model
    
    def forward(self, x):
        """"Forward pass of the MLP to get the predicted price of a renting."""
        last_input = x
        
        for i, layer in enumerate(self.layers[:-1]):
            last_input = self.dropout(self.activation(self.batch_norm[i](layer(last_input))))
        
        output = self.layers[-1](last_input)

        return output
    
if __name__ == "__main__":
    # Example usage
    model = MLP(input_dim=546, output_dim=1, num_hidden_layers=4, hidden_dimensions=[136, 34, 8, 2])
    output = model(torch.randn(32, 546))
    print(output.shape)
    print(f"Example of output: {output[0].item()}")