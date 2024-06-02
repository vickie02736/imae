import sys
sys.path.append(".")

import torch
import torch.nn as nn


class VisionTransformer(nn.Module): 

    def __init__(self, database, channel_num, image_len, patch_len, num_layers=6, nhead=8):

        super().__init__()

        self.channel_num = channel_num
        self.patch_len = patch_len
        self.image_len = image_len

        self.side_patch_num  = image_len//patch_len
        self.patch_embedding_num = self.side_patch_num**2
        self.patch_embedding_len = channel_num*patch_len*patch_len

        self.start_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding_len))
        self.end_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding_len)) 
        self.pos_embedding = nn.Parameter(torch.randn(self.patch_embedding_num+2, self.patch_embedding_len))*0.02

        self.random_tensor = torch.randn(self.channel_num,self.image_len,self.image_len) # for random masking
        
        self.nhead=nhead
        transform_layer = nn.TransformerEncoderLayer(d_model=self.patch_embedding_len, nhead=self.nhead, dropout=0.0, batch_first=True)
        self.num_layers = num_layers
        self.transformer = nn.TransformerEncoder(transform_layer, num_layers=self.num_layers)

        norm_layer=nn.LayerNorm
        self.norm = norm_layer(self.patch_embedding_len)

        self.seq_patchify = torch.vmap(self.patchify)
        self.seq_unpatchify = torch.vmap(self.unpatchify)

        self.batch_encoder = torch.vmap(self.encoder)
        self.batch_decoder = torch.vmap(self.decoder)

        self.database = database
        if self.database == 'shallow_water':
            self.conv = nn.Conv2d(channel_num, channel_num, kernel_size = 3, padding =1)
            self.seq_conv = torch.vmap(self.conv)
        elif self.database == 'moving_mnist':
            self.conv1 = nn.Conv2d(channel_num, 3*channel_num, kernel_size = 3, padding =1)
            self.conv2 = nn.Conv2d(3*channel_num, channel_num, kernel_size = 3, padding =1)
            self.seq_conv1 = torch.vmap(self.conv1)
            self.seq_conv2 = torch.vmap(self.conv2)
            self.sigmoid = nn.Sigmoid()
        else: 
            pass


    def forward(self, x): 
        x = self.batch_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.batch_decoder(x)

        # conv
        x = x.permute(1,0,2,3,4)
        if self.database == 'shallow_water':
            x = self.seq_conv(x)
        elif self.database == 'moving_mnist':
            x = self.seq_conv1(x)
            x = x.float()
            x = self.seq_conv2(x)
            x = self.sigmoid(x)
        else:
            pass
        x = x.permute(1,0,2,3,4)
        return x
 
 
    def patchify(self, x): 
        x = x.unfold(1, self.patch_len, self.patch_len).unfold(2, self.patch_len, self.patch_len)
        x = x.permute(1, 2, 0, 3, 4)
        x = x.reshape(-1, self. channel_num, self.patch_len, self.patch_len)
        x = x.reshape(self.patch_embedding_num, -1)
        return x
    

    def unpatchify(self, x): 
        x = x.view(self.side_patch_num, self.side_patch_num, self.channel_num, self.patch_len, self.patch_len)
        x = x.permute(2, 0, 3, 1, 4).reshape(self.channel_num, self.image_len, self.image_len)
        return x
    

    def encoder(self, x): 
        x = self.seq_patchify(x)
        start_embeddings = self.start_embedding.repeat(x.shape[0], 1, 1)
        end_embeddings = self.end_embedding.repeat(x.shape[0], 1, 1)
        x = torch.cat((start_embeddings, x, end_embeddings), 1) # add start and end tokens
        pos_embeddings = self.pos_embedding.repeat(x.shape[0], 1, 1).to(x.device)
        x += pos_embeddings # add positional embeddings
        x = x.view(-1, self.patch_embedding_len)
        return x
    
    
    def decoder(self, x): 
        x = x.unsqueeze(0)
        x = x.view(-1, self.patch_embedding_num+2, self.patch_embedding_len)
        x = x[:, 1:-1, :] # remove start and end tokens

        x = self.seq_unpatchify(x)
        return x



'''
https://github.com/ndrplz/ConvLSTM_pytorch
'''
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # return layer_output_list, last_state_list
        return layer_output_list[0]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# import yaml
# config = yaml.load(open("../configs/convlstm_sw_linear.yaml", "r"), Loader=yaml.FullLoader)
# model = ConvLSTM(
#     input_dim=config['channels'], 
#     hidden_dim=config['model']['hidden_dim'], 
#     kernel_size=tuple(config['model']['kernel_size']), 
#     num_layers=config['model']['num_layers'], 
#     batch_first=True, 
#     bias=True, 
#     return_all_layers=False
# )
# x = torch.randn(2, 10, 3, 128, 128)
# y = model(x)
# print(y.shape)



class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dim, input_channels=3):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim

        #initialize encoder and decoder
        self.encode = nn.Sequential(
                    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # Downsamples to 64x64 if input is 128x128
                    
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
                    
                    nn.Conv2d(1024, self.latent_dim, kernel_size=4, stride=1, padding=0),  # Downsamples to 1x1
                    nn.BatchNorm2d(self.latent_dim),
                    nn.GELU()
                )
        self.decode = nn.Sequential(  
                    nn.ConvTranspose2d(self.latent_dim, 1024, kernel_size=4, stride=1, padding=0),  # Upscales to 4x4
                    nn.BatchNorm2d(1024),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscales to 8x8
                    nn.BatchNorm2d(512),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscales to 16x16
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscales to 32x32
                    nn.BatchNorm2d(128),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscales to 64x64
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                    
                    nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upscales to 128x128
                )
        # output_padding parameter is used in some layers to ensure that the spatial dimensions are correctly upscaled, 
        # especially when the stride is more than 1

        # Batch normalization for the encoder's output (latent code)
        self.bn = nn.BatchNorm2d(self.latent_dim, affine=False)

    def forward(self, x):
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

    def __init__(self, input_size, hidden_size):
        
        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_size = input_size, # The number of expected features in the input.
                            hidden_size = hidden_size, # The number of features in the hidden state.
                            num_layers = 5, # The number of recurrent layers.
                            dropout = 0.5, # Dropout probability for each LSTM layer except the last one
                            batch_first = True)

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