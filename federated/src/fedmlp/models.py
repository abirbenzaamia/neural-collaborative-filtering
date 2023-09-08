from torch.nn import functional as F
from typing import Optional, Tuple
from torch import Tensor
import torch

from fedmlp.config import MODEL_PARAMETERS, DEVICE


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        params = MODEL_PARAMETERS['FedMLP']
        layers = params['layers']
        mlp_dim = int(layers[0] / 2)
        #print('-----------------', mlp_dim)
        self.mlp_embedding_user = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_dim, device=DEVICE)
        self.mlp_embedding_item = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_dim, device=DEVICE)
        #print('-----------------', mlp_dim)
        self.mlp_embedding_user.weight.data.uniform_(0, 1)
        self.mlp_embedding_item.weight.data.uniform_(0, 1)
        self.mlp = torch.nn.ModuleList()
        current_dim = 48
        for idx in range(1, len(layers)):
            self.mlp.append(torch.nn.Linear(current_dim, layers[idx]))
            current_dim = layers[idx]
            self.mlp.append(torch.nn.ReLU())
        self.output_layer = torch.nn.Linear(in_features=layers[-1], out_features=1, device=DEVICE)
    def forward(self, user_input: Tensor,
                item_input: Tensor,
                target: Optional[Tensor] = None) -> Tuple[Tensor, Optional[float]]:
        # mlp   
        mlp_user_latent = torch.nn.Flatten()(self.mlp_embedding_user(user_input))
        mlp_item_latent = torch.nn.Flatten()(self.mlp_embedding_item(item_input))
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=1)

        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)
        predict_vector = mlp_vector
        logits = self.output_layer(predict_vector)
        loss = None
        if target is not None:
            target = target.view(target.shape[0], 1).to(torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, target)

        #logits = torch.nn.Linear(logits)
        #print(logits)
        #print(logits.shape)
        return logits, loss
