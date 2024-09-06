import torch
import torch.nn as nn
import numpy as np

# def symm(X):
#     return 0.25*(np.rot90(X,2).T + np.rot90(X,2) + X + X.T)


# class Band_generator(nn.Module):
#     def __init__(self, latent_dim=30):
#         super(self.__class__, self).__init__()
#         self.latent_dim = latent_dim
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [16, 5, 5]
#             nn.Tanh(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Larger layer [32, 3, 3]
#             nn.Tanh(),
#             nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),  # Reducing to 8 channels [8, 3, 3]
#             nn.Tanh()
#         )
#         # self.flatten = nn.Flatten()
#         # self.fc1 = nn.Linear(8 * 3 * 3, latent_dim)

#         # Decoder
#         self.fc2 = nn.Linear(latent_dim, 8 * 3 * 3)
#         self.unflatten = nn.Unflatten(1, (8, 3, 3))
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1),  # Larger layer [32, 5, 5]
#             nn.Tanh(),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # [16, 5, 5]
#             nn.Tanh(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [1, 10, 10]
#             # nn.Tanh()  # Sigmoid for values between 0 and 1
#         )
#         self.encode_mu = torch.nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(8 * 3 * 3, latent_dim),
#             torch.nn.ReLU()
#         )

#         self.encode_logvar = torch.nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(8 * 3 * 3, latent_dim)
#         )
        

#     def encode(self, x):
#         x = self.encoder(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         return x

#     def decode(self, z):
#         z = self.fc2(z)
#         z = self.unflatten(z)
#         z = self.decoder(z)
#         return z

#     def reparameterise(self, mu, logvar):
#         sigma = torch.exp(0.5*logvar)
#         eps = torch.randn_like(sigma)
#         return mu + eps * sigma
    
#     def forward(self,x):    
#         encoded = self.encoder(x)
#         self.mu, self.log_var = self.encode_mu(encoded), self.encode_logvar(encoded)
#         z = self.reparameterise(self.mu, self.log_var)
#         y = self.decode(z)
#         return y

#     def load_checkpoint(self, checkpoint_path):
#         state = torch.load(checkpoint_path)
#         self.load_state_dict(state['state_dict'])
#         print('model loaded from %s' % checkpoint_path)

#     def getBS(self):
#         random_vector = torch.randn(1, 30)

#         # Decode the random vector to get an output image
#         with torch.no_grad():
#             random_decoded_output = self.decode(random_vector).cpu().detach().numpy()
#         return symm(random_decoded_output[0,0])
    


class Band_generator(nn.Module):
    def __init__(self, latent_dim=7):
        super(self.__class__, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,latent_dim),
            torch.nn.ReLU()
        )
        
        # Decoder

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,100),
            nn.Unflatten(1, (1, 10, 10))

        )


    def decode(self, z):
        z = self.decoder(z)
        z = (z + torch.transpose(z, 2, 3) )*0.5
        z = (z + torch.rot90( 
                torch.transpose( 
                                torch.rot90(z,1, [2, 3]), 2, 3
                                ),
                                3, [2, 3])
            )*0.5
        return z


    def forward(self,x):    
        z = self.encoder(x)
        
        y = self.decode(z)
        # print(y.shape)
        
        return y

    def getBS(self):
        random_vector = torch.randn(1, 7)

        # Decode the random vector to get an output image
        with torch.no_grad():
            random_decoded_output = self.decode(random_vector).cpu().detach().numpy()
        return random_decoded_output[0,0]

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.load_state_dict(state['state_dict'])
        print('model loaded from %s' % checkpoint_path)
