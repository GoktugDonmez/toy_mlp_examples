import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)





# dataset

def generate_sine_data(n:int,x_min:float ,x_max: float ,sigma: float = 0.05):
    x = torch.linspace(x_min,x_max,n).unsqueeze(1)
    ## unsqueeze to add a dimension from (n,) list of n to (n,1) list of n 1-dimensional vectors

    y_clean = torch.sin(5*x)

    y = y_clean + torch.randn_like(y_clean) * sigma

    return x,y


## init class

class ThetaMLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super().__init__()

        # Manually define weights and biases as parameters to start 
        #self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        #self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        #self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        #self.b2 = nn.Parameter(torch.zeros(hidden_dim))

        #self.W3 = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        #self.b3 = nn.Parameter(torch.zeros(output_dim))

        ## built in 
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )




    def forward(self,x):
        #z1 = x @ self.W1.T + self.b1
        #h1 = torch.relu(z1)

        #z2 = h1 @ self.W2.T + self.b2
        #h2 = torch.relu(z2)

        #output layer
        #z3 = h2 @ self.W3.T + self.b3

        #return z3
        return self.net(x)


def main():

    N = 1000
    sigma = 0.05
    x_min = -math.pi
    x_max = math.pi

    x,y = generate_sine_data(N,x_min,x_max,sigma)
    ## scale the input 


    hidden_dim = 50
    lr = 1e-3
    epochs = 2000

    model = ThetaMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    criterion = nn.MSELoss() ## why is this regresion loss look on other cases too

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):
        model.train()

        y_pred = model(x)
        loss = criterion(y_pred,y)


        #model.zero_grad() # equivalent to below
        #for param in model.parameters():
        #    if param.grad is not None:
        #        param.grad.zero_()
        optimizer.zero_grad()


        loss.backward() # backpropagate


        optimizer.step()
        #with torch.no_grad():
        #    for param in model.parameters():
        #        param = param - lr * param.grad

        ### GRADIENT CHECKING 
        if epoch % 100 == 0:
            with torch.no_grad():
                #  gradient norms
                total_norm = 0.0
                #for p in model.parameters():
                #    total_norm += p.grad.norm().item() ** 2
                #total_norm = total_norm ** 0.5


            print(f"Epoch {epoch:04d} | loss={loss.item():.4f}")
                  #| grad‖·‖={total_norm:.4e} ")
            

    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(x_min, x_max, 1000).unsqueeze(1)
        y_plot_pred = model(x_plot)
        y_plot_true = torch.sin(5 * x_plot)

    plt.figure(figsize=(10, 6))
    plt.scatter(x.numpy(), y.numpy(), s=10, alpha=0.3, label="Noisy data")
    plt.plot(x_plot.numpy(), y_plot_true.numpy(), label="True sin(5x)", linewidth=2)
    plt.plot(x_plot.numpy(), y_plot_pred.numpy(), '--', label="MLP prediction", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.title("SGD: MLP learns sin(5x)")
    plt.show()

    




if __name__ == "__main__":
    main()




