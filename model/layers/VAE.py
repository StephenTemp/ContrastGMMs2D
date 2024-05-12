import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

'''
Credit and tutorial to https://avandekleut.github.io/vae/ for code and
descriptions
'''


'''
ENCODER: The encoder learns a non-linear transformation e : X → Z
that projects the data from the original high-dimensional input space 
X to a lower-dimensional latent space Z. We call z = e(x), a latent vector. 
A latent vector is a low-dimensional representation of a data point that 
contains information about x. The transformation e should have certain properties, 
like similar values of x should have similar latent vectors (and dissimilar values of 
x should have dissimilar latent vectors).
'''
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)
    
'''
DECODER: A decoder learns a non-linear transformation d : Z → X
that projects the latent vectors back into the original high-dimensional input space 
X. This transformation should take the latent vector z = e(x)and reconstruct the 
original input data ^ x = d(z) = d(e(x)).
'''
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))
    
'''
AUTOENCONDER: An autoencoder is just the composition of the encoder and the decoder f(x) = d(e(x)). 
The autoencoder is trained to minimize the difference between the input x and the reconstruction 
^x using a kind of reconstruction loss. Because the autoencoder is trained as a whole 
(we say it’s trained “end-to-end”), we simultaneosly optimize the encoder and the decoder.
'''
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)