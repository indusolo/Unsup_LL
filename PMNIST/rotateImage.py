### Variational Autoencoder
import torch
import torch.nn as nn

# VAE Model Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' #torch.device("cuda" if cuda else "cpu")
print(DEVICE)
num_classes = 10
batch_size = 60

x_dim  = 784
hidden_dim1 = 650
hidden_dim2 = 300
latent_dim = 32

lr = 1e-3
epochs = 100
# train_on_gpu = torch.cuda.is_available()
# print(train_on_gpu)

# Encoder

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim1)
        self.FC_input2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.FC_input3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.FC_mean  = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim2, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        # h_       = self.LeakyReLU(self.FC_input3(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     
                                                      
        
        return mean, log_var

# Decoder

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim1,hidden_dim2, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim2)
        self.FC_hidden2 = nn.Linear(hidden_dim2, hidden_dim1)
        # self.FC_hidden3 = nn.Linear(hidden_dim1, hidden_dim1)
        self.FC_output = nn.Linear(hidden_dim1, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        # h     = self.LeakyReLU(self.FC_hidden3(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

encoder = Encoder(input_dim=x_dim, hidden_dim1=hidden_dim1,hidden_dim2=hidden_dim2, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim1 = hidden_dim1,hidden_dim2=hidden_dim2, output_dim = x_dim)

