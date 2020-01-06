import torch
from torchfm.layer import FeaturesEmbedding, MultiLayerPerceptron
from int_module import InterpretModule


class NeuralCollaborativeFiltering(torch.nn.Module):
    """
    A pytorch implementation of Neural Collaborative Filtering.

    Reference:
        X He, et al. Neural Collaborative Filtering, 2017.
    """

    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        x = self.mlp(x.view(-1, self.embed_output_dim))
        gmf = user_x * item_x
        x = torch.cat([gmf, x], dim=1)
        y = self.fc(x).squeeze(1)
        return torch.sigmoid(y), x # output prediction + last layer


class InterpretableModel(torch.nn.Module):
    def __init__(self, model, int_module):
        """
        :param model:  a non-interpretable model
        :param int_module: the interpret module InterpretModule
        """
        super().__init__()
        assert isinstance(int_module, InterpretModule)
        self.model = model
        self.int_module = int_module

    def forward(self, x, y):
        """
        :param x: [batch, 2], user, item
        :param y: genre input [batch, num_asp]
        """
        prediction, feature = self.model(x)
        feature_nograd = feature.detach()
        int_loss = self.int_module(y, feature_nograd)
        int_loss = int_loss.sum() / x.shape[0]
        # print(int_loss)
        return prediction, int_loss

    def importance(self, x, y):
        """
        get importance of each aspect 
        :param x: [batch, 2], user, item
        :param y: genre input [batch, num_asp]
        return: importance of each aspect [batch, num_asp]
        """
        _, feature = self.model(x)
        feature_nograd = feature.detach()
        return self.int_module.importance(y, feature_nograd)


