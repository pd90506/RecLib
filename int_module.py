import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretModule(torch.nn.Module):
    def __init__(self, num_asp, dim_emb):
        """
        :param num_asp: number of aspects used to interpret
        :param dim_emb: the dimension of the aspect latent space
        """
        super(InterpretModule, self).__init__()
        self.asp_emb = Aspect_emb(num_asp, dim_emb)
    
    def forward(self, x, y):
        """
        :param x: aspect multihot vector, [batch_size, num_asp]
        :param y: final hiden layer of original model, [batch, dim_emb]
        return: interpret loss L_int
        """
        att_vector = self.importance(x, y)
        x = self.asp_emb(x) # [batch_size, num_asp, dim_emb]
        y = y.unsqueeze(-1) # [batch_size, dim_emb, 1]
        # calculate reconstructed vector \tilde{u}
        tilde_u = torch.bmm(att_vector.unsqueeze(1), x).squeeze(1)#.squeeze(1) # [batch_size, dim_emb]
        # print(tilde_u.shape)
        y = y.squeeze(-1)
        out = nn.functional.pairwise_distance(y, tilde_u) # [batch_size, 1]
        # print(out.shape)
        return out

    # def forward(self, x, y):
    #     """
    #     :param x: aspect multihot vector, [batch_size, num_asp]
    #     :param y: final hiden layer of original model, [batch, dim_emb]
    #     return: interpret loss L_int
    #     """
    #     x = self.asp_emb(x) # [batch_size, num_asp, dim_emb]
    #     y = y.unsqueeze(-1) # [batch_size, dim_emb, 1]
    #     # attention obtained by inner product
    #     att_vector = torch.bmm(x, y).squeeze(-1) # [batch_size, num_asp]
    #     # calculate reconstructed vector \tilde{u}
    #     tilde_u = torch.bmm(att_vector.unsqueeze(1), x).squeeze(1)#.squeeze(1) # [batch_size, dim_emb]
    #     # print(tilde_u.shape)
    #     y = y.squeeze(-1)
    #     out = nn.functional.pairwise_distance(y, tilde_u) # [batch_size, 1]
    #     # print(out.shape)
    #     return out

    def importance(self, x, y):
        """
        :param x: aspect multihot vector, [batch_size, num_asp]
        :param y: final hiden layer of original model, [batch, dim_emb]
        return: attention [batch, num_asp]
        """
        x = self.asp_emb(x) # [batch_size, num_asp, dim_emb]
        y = y.unsqueeze(-1) # [batch_size, dim_emb, 1]
        # attention obtained by inner product
        att_vector = torch.bmm(x, y).squeeze(-1) # [batch_size, num_asp]
        return att_vector





class Aspect_emb(nn.Module):
    """
    module to embed each aspect to the latent space.
    """
    def __init__(self, num_asp, e_dim):
        super(Aspect_emb, self).__init__()
        self.num_asp = num_asp
        self.W = nn.Parameter(torch.randn(num_asp, e_dim)) # contains all aspect latents

    def forward(self, x):
        # note that x  are multi-hot vectors of dimension [batch_size, num_asp]
        # x = x.reshape([x.shape[0], x.shape[1], 1])
        x = torch.unsqueeze(x, -1)
        x = x.expand(-1, -1, self.W.shape[1])
        asp_latent = torch.mul(x, self.W) # [batch_size, num_asp, dim_emb]
        # we must normalize asp_latent per aspect??? not necessary
        asp_latent = F.normalize(asp_latent, p=2, dim=2)
        

        return asp_latent