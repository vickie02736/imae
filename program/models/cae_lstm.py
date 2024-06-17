import torch
import torch.nn as nn



class ConvAutoencoder(nn.Module):

    def __init__(self, config):
        super(ConvAutoencoder, self).__init__()

        self.latent_dim = config['cae']['latent_dim']
        input_channels = config['channels']

        #initialize encoder and decoder
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # Downsamples to 64x64 if input is 128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples to 32x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples to 16x16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),  # Downsamples to 8x8
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),  # Downsamples to 4x4
            nn.Conv2d(1024,
                      self.latent_dim,
                      kernel_size=4,
                      stride=1,
                      padding=0),  # Downsamples to 1x1
            nn.BatchNorm2d(self.latent_dim),
            nn.GELU())
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim,
                               1024,
                               kernel_size=4,
                               stride=1,
                               padding=0),  # Upscales to 4x4
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024,
                               512,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Upscales to 8x8
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.ConvTranspose2d(512,
                               256,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Upscales to 16x16
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256,
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Upscales to 32x32
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Upscales to 64x64
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64,
                               input_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Upscales to 128x128
        )

        # Batch normalization for the encoder's output (latent code)
        self.bn = nn.BatchNorm2d(self.latent_dim, affine=False)

    def forward(self, x, sequence_input=False):
        if sequence_input:
            batch_size, sequence_length, channels, height, width = x.shape
            x = x.view(batch_size * sequence_length, channels, height, width)
            latent_code = self.encoder(x)
            latent_code = latent_code.view(batch_size, sequence_length, -1)
            x_hat = self.decoder(latent_code.view(batch_size * sequence_length, -1))
            x_hat = x_hat.view(batch_size, sequence_length, channels, height, width)
        else:
            latent_code = self.encoder(x)
            x_hat = self.decoder(latent_code)

        return {'latent_code': latent_code, 'x_hat': x_hat}

    def encoder(self, x):
        x = self.encode(x)
        x = self.bn(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def decoder(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decode(x)
        return x


# model = ConvAutoencoder(latent_dim=64, input_channels=3)
# x = torch.randn(2, 3, 128, 128)
# latent = model.encoder(x)
# y = model.decoder(latent)
# print(latent.shape, y.shape)
# print(y['latent_code'].shape, y['x_hat'].shape)


class LSTMPredictor(nn.Module):

    def __init__(self, config):

        super(LSTMPredictor, self).__init__()

        input_size = config['cae']['latent_dim']
        hidden_size = config['cae_lstm']['hidden_dim']

        self.lstm = nn.LSTM(
            input_size=
            input_size,  # The number of expected features in the input.
            hidden_size=
            hidden_size,  # The number of features in the hidden state.
            num_layers=5,  # The number of recurrent layers.
            dropout=
            0.5,  # Dropout probability for each LSTM layer except the last one
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


# x = torch.randn(2, 10, 64)
# model = LSTMPredictor(input_size=64, hidden_size=640)
# y = model(x)
# print(y.shape)
