import sys
sys.path.append(".")

import torch
import torch.nn as nn


class VisionTransformer(nn.Module): 

    def __init__(self, channel_num, image_len, patch_len, device_id, nhead=6):

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
        self.pos_embedding = self.pos_embedding.to(device_id)

        self.random_tensor = torch.randn(self.channel_num,self.image_len,self.image_len).to(device_id) # for random masking
        
        self.nhead=nhead
        transform_layer = nn.TransformerEncoderLayer(d_model=self.patch_embedding_len + self.nhead, nhead=self.nhead, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(transform_layer, num_layers=6)

        norm_layer=nn.LayerNorm
        self.norm = norm_layer(self.patch_embedding_len + self.nhead)

        self.batch_mask = torch.vmap(self.mask)

        self.seq_patchify = torch.vmap(self.patchify)
        self.seq_unpatchify = torch.vmap(self.unpatchify)

        self.batch_encoder = torch.vmap(self.encoder, in_dims=(0, None))
        self.batch_decoder = torch.vmap(self.decoder)

        self.conv = nn.Conv2d(3, 3, kernel_size = 3, padding =1)
        self.seq_conv = torch.vmap(self.conv)


    def forward(self, x, num_mask): 
        num_mask = 5

        if num_mask != 0:

            weights = torch.ones(x.shape[1]).expand(x.shape[0], -1)
            idx = torch.multinomial(weights, num_mask, replacement=False).to(x.device)

            # encode
            x = self.batch_encoder(x, idx)
        else: 
            # encode
            x = self.batch_encoder(x, idx=None)

        # # transformer
        x = self.transformer(x)
        x = self.norm(x)

        # # decode
        x = self.batch_decoder(x)

        # conv
        x = x.permute(1,0,2,3,4)
        x = self.seq_conv(x)
        x = x.permute(1,0,2,3,4)

        return x
 
    def patchify(self, x): 
        # Unfold the height and width dimensions
        x = x.unfold(1, self.patch_len, self.patch_len).unfold(2, self.patch_len, self.patch_len)

        # Reshape the unfolded dimensions to get the patches 
        x = x.permute(1, 2, 0, 3, 4)
        x = x.reshape(-1, self. channel_num, self.patch_len, self.patch_len)
        x = x.reshape(self.patch_embedding_num, -1)
        return x
    
    def unpatchify(self, x): 
        x = x.view(self.side_patch_num, self.side_patch_num, self.channel_num, self.patch_len, self.patch_len)
        x = x.permute(2, 0, 3, 1, 4).reshape(self.channel_num, self.image_len, self.image_len)
        return x
    
    def encoder(self, x, idx=None): 
        # apply patchify to the sequence
        x = self.seq_patchify(x)

        # add start and end tokens
        start_embeddings = self.start_embedding.repeat(x.shape[0], 1, 1)
        end_embeddings = self.end_embedding.repeat(x.shape[0], 1, 1)
        x = torch.cat((start_embeddings, x, end_embeddings), 1)

        # add positional embeddings
        pos_embeddings = self.pos_embedding.repeat(x.shape[0], 1, 1)
        x += pos_embeddings # [10, 66, 768]

        # add binary label for masking
        label = torch.ones(x.shape[0], x.shape[1], self.nhead).to(x.device)
        if idx is not None:
            label[idx] = 0
        x = torch.cat((x, label), dim=2) # [10, 66, 769]

        # pass through the transformer
        x = x.view(-1, self.patch_embedding_len + self.nhead) # + self.head is label
        return x
    
    def decoder(self, x): 
        x = x.unsqueeze(0)
        x = x.view(-1, self.patch_embedding_num+2, self.patch_embedding_len+self.nhead)

        # remove label
        x = x[:, :, :-self.nhead]
        # remove start and end tokens
        x = x[:, 1:65, :]
        # apply unpatchify to the sequence
        x = self.seq_unpatchify(x)
        return x
    
    def mask(self, x, idx): 
        # mask the input tensor
        self.random_tensor = self.random_tensor.to(x.device)
        x[idx] = self.random_tensor
        return x