import torch
import torch.nn as nn



class LSTMModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        seq_pred = self.linear(lstm_out)
        return seq_pred
    


class ConvAutoencoder128(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder128, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # Output: 16 x 64 x 64 (or 16 x 32 x 32 if input is 3 x 64 x 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 32 x 32 x 32 (or 32 x 16 x 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 x 16 x 16 (or 64 x 8 x 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # Output: 128 x 8 x 8 (or 128 x 4 x 4)
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # Output: 256 x 4 x 4 (or 256 x 2 x 2)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4, latent_dim)  # Adjust size according to output from last conv layer
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*4*4),  # Adjust size to match the output of the encoder
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),    # Start unflattening here; change dimensions if needed
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # Output: 128 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # Output: 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # Output: 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),    # Output: 16 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),     # Output: 3 x 128 x 128
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    


class ConvAutoencoder64(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder64, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # Output: 128 x 4 x 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, latent_dim)  # Flatten and connect to a latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),  # Start from the latent dimension
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),    # Unflatten to the size before the last conv layer
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # Output: 64 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # Output: 1 x 64 x 64
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded