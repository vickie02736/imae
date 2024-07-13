import torch
import torch.nn as nn


class VisionTransformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.database = config['database']
        self.channel_num, self.image_h, self.image_w = config['image_size']
        self.patch_h, self.patch_w = config['imae']['patch_size']
        self.num_layers = config['imae']['num_layers']
        self.nhead = config['imae']['nhead']

        # self.side_patch_num = self.image_len // self.patch_len
        self.patch_num_h = self.image_h // self.patch_h
        self.patch_num_w = self.image_w // self.patch_w
        self.patch_embedding_num = self.patch_num_h * self.patch_num_w
        self.patch_embedding_len = self.channel_num * self.patch_h * self.patch_w

        self.start_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding_len))
        self.end_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding_len))
        self.pos_embedding = nn.Parameter(
            torch.randn(self.patch_embedding_num + 2,
                        self.patch_embedding_len)) * 0.02

        self.random_tensor = torch.randn(self.channel_num, self.image_h,
                                         self.image_w)  # for random masking

        transform_layer = nn.TransformerEncoderLayer(
            d_model=self.patch_embedding_len,
            nhead=self.nhead,
            dropout=0.0,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(transform_layer,
                                                 num_layers=self.num_layers)

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(self.patch_embedding_len)

        self.seq_patchify = torch.vmap(self.patchify)
        self.seq_unpatchify = torch.vmap(self.unpatchify)

        self.batch_encoder = torch.vmap(self.encoder)
        self.batch_decoder = torch.vmap(self.decoder)

        if self.database == 'shallow_water':
            self.conv = nn.Conv2d(self.channel_num,
                                  self.channel_num,
                                  kernel_size=3,
                                  padding=1)
            self.seq_conv = torch.vmap(self.conv)
        elif self.database == 'moving_mnist':
            self.conv1 = nn.Conv2d(self.channel_num,
                                   3 * self.channel_num,
                                   kernel_size=3,
                                   padding=1)
            self.conv2 = nn.Conv2d(3 * self.channel_num,
                                   self.channel_num,
                                   kernel_size=3,
                                   padding=1)
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
        x = x.permute(1, 0, 2, 3, 4)
        if self.database == 'shallow_water':
            x = self.seq_conv(x)
        elif self.database == 'moving_mnist':
            x = self.seq_conv1(x)
            x = x.float()
            x = self.seq_conv2(x)
            x = self.sigmoid(x)
        else:
            pass
        x = x.permute(1, 0, 2, 3, 4)
        return x

    def patchify(self, x):
        x = x.unfold(1, self.patch_h, self.patch_w).unfold(2, self.patch_h, self.patch_w)
        x = x.permute(1, 2, 0, 3, 4)
        x = x.reshape(-1, self.channel_num, self.patch_h, self.patch_w)
        x = x.reshape(self.patch_embedding_num, -1)
        return x

    def unpatchify(self, x):
        x = x.view(self.patch_num_h, self.patch_num_w, self.channel_num,
                   self.patch_h, self.patch_w)
        x = x.permute(2, 0, 3, 1, 4).reshape(self.channel_num, self.image_h,
                                             self.image_w)
        return x

    def encoder(self, x):
        x = self.seq_patchify(x)
        start_embeddings = self.start_embedding.repeat(x.shape[0], 1, 1)
        end_embeddings = self.end_embedding.repeat(x.shape[0], 1, 1)
        x = torch.cat((start_embeddings, x, end_embeddings),
                      1)  # add start and end tokens
        pos_embeddings = self.pos_embedding.repeat(x.shape[0], 1,
                                                   1).to(x.device)
        x += pos_embeddings  # add positional embeddings
        x = x.view(-1, self.patch_embedding_len)
        return x

    def decoder(self, x):
        x = x.unsqueeze(0)
        x = x.view(-1, self.patch_embedding_num + 2, self.patch_embedding_len)
        x = x[:, 1:-1, :]  # remove start and end tokens

        x = self.seq_unpatchify(x)
        return x


# import yaml
# config = yaml.load(open("/home/uceckz0/Project/imae/configs/example_config.yaml", "r"), Loader=yaml.FullLoader)
# model = VisionTransformer(config)
# x = torch.randn(2, 10, 3, 64, 128)
# y = model(x)
# print(y.shape)