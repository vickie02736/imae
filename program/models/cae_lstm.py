import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, config):
        super(ConvAutoencoder, self).__init__()
        input_channels, image_height, image_width = config['image_size'] 
        latent_dim = config['cae']['latent_dim']
        
        # Encoder
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # (batch, C, H, W) -> (batch, 16, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (batch, 16, H/2, W/2) -> (batch, 32, H/4, W/4)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (batch, 32, H/4, W/4) -> (batch, 64, H/8, W/8)
            nn.ReLU(True),
            nn.Flatten()                                           # (batch, 64, H/8, W/8) -> (batch, 64*H/8*W/8)
        )
        
        self.fc1 = nn.Linear(64 * (image_height // 8) * (image_width // 8), latent_dim)  # (batch, 64*H/8*W/8) -> (batch, latent_dim)
        
        self.fc2 = nn.Linear(latent_dim, 64 * (image_height // 8) * (image_width // 8))  # (batch, latent_dim) -> (batch, 64*H/8*W/8)
        
        # Decoder
        self.decode = nn.Sequential(
            nn.Unflatten(1, (64, image_height // 8, image_width // 8)),          # (batch, 64*H/8*W/8) -> (batch, 64, H/8, W/8)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (batch, 64, H/8, W/8) -> (batch, 32, H/4, W/4)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (batch, 32, H/4, W/4) -> (batch, 16, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # (batch, 16, H/2, W/2) -> (batch, C, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        if x.dim() == 5:  # (batch_size, sequence_length, input_channels, image_height, image_width)
            batch_size, sequence_length, _, _, _ = x.size()
            x = x.view(batch_size * sequence_length, x.size(2), x.size(3), x.size(4))
            x = self.encode(x)
            x = self.fc1(x)
            x = x.view(batch_size, sequence_length, -1)  # (batch_size, sequence_length, latent_dim)
        else:  # (batch_size, input_channels, image_height, image_width)
            x = self.encode(x)
            x = self.fc1(x)  # (batch_size, latent_dim)
        return x

    def decoder(self, x):
        if x.dim() == 3:  # (batch_size, sequence_length, latent_dim)
            batch_size, sequence_length, _ = x.size()
            x = x.view(batch_size * sequence_length, -1)
            x = self.fc2(x)
            x = self.decode(x)
            x = x.view(batch_size, sequence_length, x.size(1), x.size(2), x.size(3))  # (batch_size, sequence_length, C, H, W)
        else:  # (batch_size, latent_dim)
            x = self.fc2(x)
            x = self.decode(x)  # (batch_size, input_channels, image_height, image_width)
        return x



class LSTMPredictor(nn.Module):

    def __init__(self, config):

        super(LSTMPredictor, self).__init__()

        input_size = config['cae']['latent_dim']
        hidden_size = config['cae_lstm']['hidden_dim']

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,  
            num_layers=5,  
            dropout=0.5,  
            batch_first=True)

        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, fragment_length, latent_dim_AE = x.shape
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        # out: (batch_size, latent_dim_AE, hidden_size)
        out = out.reshape(batch_size, fragment_length, latent_dim_AE)
        return out