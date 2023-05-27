import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim # test 
        self.hidden_dim = hidden_dim * 2
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim) # Output mean and log std dev
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Regression loss metric
        self.mse_loss_fun = nn.MSELoss(reduction='mean')
        
    def encode(self, x):
        # Compute mean and log std dev
        h = self.encoder(x)
        mu, log_std = torch.chunk(h, 2, dim=-1)
        
        # Sample from Gaussian distribution
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        
        return z, mu, log_std
    
    def decode(self, z):
        # Compute predicted output
        h = self.decoder(z)
        x_hat = torch.sigmoid(h)
        
        return x_hat
    
    def forward(self, x):
        # Encode input into latent space
        z, mu, log_std = self.encode(x)
        
        # Decode latent point into predicted output
        x_hat = self.decode(z)
        
        # Compute MSE loss
        self.mse_loss_val = self.mse_loss_fun(x_hat, x)
        
        return x_hat, mu, log_std, self.mse_loss_val
    
    def loss(self, x, x_hat, mu, log_std, mse_loss_weight=1.0):
        # Compute reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        
        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())
        
        # Compute total loss
        total_loss = recon_loss + kl_loss + mse_loss_weight * self.mse_loss_val
        # total_loss = recon_loss + kl_loss + mse_loss_weight * self.mse_loss
        
        return total_loss
    

# class PSPR():
#     def __init__(self,):
#     Z = np.zeros([2,1])
#     X_hat = np.zeros(X0.shape)
#     i = 0
#     for x_sample in X0:
#     # x_sample = X0[2]
#         x_sample_device = torch.Tensor(x_sample).to(my_device) 
#         z, mu, log_std = vae.encode(x_sample_device)
#         z_np = z.detach().cpu().numpy().reshape([2,1])
#         x_hat, mu, log_std, mse_loss = vae(x_sample_device)
#         Z = np.append(Z, z_np, axis = 1)  
#         X_hat[i,:] = x_hat.detach().cpu().numpy()
#         i +=1
#         # break
#     Zf = np.delete(Z, 0,1) # Z final witn initial zeros delete 
#     Xnorm = np.sum(np.abs(X0)**2,axis=-1)**(1./2)


class wizal():
    def __init__(slef,): pass
