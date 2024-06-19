from torch import save, load, concat
from torch import clamp as t_clamp
from torch import nn, no_grad, tensor, matmul
from torch import sum as t_sum
from torch import max as t_max
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import json
from torch import Tensor, einsum, float32, nn, sqrt, tensor, ones_like


    
    
class Mean(nn.Module):
    """Mean pooling-based user model."""

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, embeddings: Tensor, history_mask: Tensor) -> Tensor:
        if history_mask is None:
            return embeddings.mean(dim=0, keepdim=True)
        
        numerators = einsum("xyz,xy->xyz", embeddings, history_mask).sum(dim=1)

        # Clamp all values in [min, max] to prevent zero division
        denominators = torch.clamp(history_mask.sum(dim=-1), min=1e-9)

        return einsum("xz,x->xz", numerators, 1 / denominators)

class ScaledDotProduct(nn.Module):
    """Scaled-Dot Product Alignment Model."""

    def __init__(self, embedding_dim: int = 312):
        super(ScaledDotProduct, self).__init__()

        self.scale = 1.0 / sqrt(tensor(embedding_dim, dtype=float32))

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        return einsum("xz,xyz->xy", Q, K).mul(self.scale)

class Attention(nn.Module):
    """Standard Attention Mechanism."""

    def __init__(
        self, embedding_dim: int = 312,
    ):
        super(Attention, self).__init__()

        self.alignment_model = ScaledDotProduct(embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, attention_mask: Tensor
    ) -> Tensor:
        # Scoring / Alignment --------------------------------------------------
        alignment_scores = self.alignment_model(Q, K)

        if attention_mask is None:
            attention_mask = ones_like(alignment_scores)
        # Masking --------------------------------------------------------------
        alignment_scores = alignment_scores.masked_fill(
            attention_mask == 0, -1e4
        )

        # Normalization --------------------------------------------------------
        attention_weights = self.softmax(alignment_scores)

        # Aggregation ----------------------------------------------------------
        return einsum("xy,xyz->xz", attention_weights, V)

class UserEncoder(nn.Module):
    """Aggregates user history to generate user embedding."""

    def __init__(
        self, aggregation_mode: str, embedding_dim: int = 312,
    ):
        super(UserEncoder, self).__init__()

        self.aggregation_mode = aggregation_mode
        self.embedding_dim = embedding_dim

        if aggregation_mode == "mean":
            self.aggregator = Mean()
        elif aggregation_mode == "attention":
            self.aggregator = Attention(embedding_dim)
        else:
            raise NotImplementedError()

    def forward(
        self,
        user_doc_embeddings: Tensor,
        query_embeddings: Tensor,
        history_mask: Tensor = None,
    ) -> Tensor:
        if self.aggregation_mode == "mean":
            return self.aggregator(user_doc_embeddings, history_mask)
        else:
            return self.aggregator(
                Q=query_embeddings,
                K=user_doc_embeddings.view(1, -1, self.embedding_dim),
                V=user_doc_embeddings.view(1, -1, self.embedding_dim),
                attention_mask=history_mask,
            )

class PersonalizationModel(nn.Module):
    def __init__(
        self,
        doc_embeddings,
        doc_id_to_index,
        query_embeddings,
        query_id_to_index,
        aggregation_mode='mean', # 'attention
        device='cpu',
    ):
        super(PersonalizationModel, self).__init__()
        
        self.device = device

        self.doc_embeddings = load(doc_embeddings)
        self.doc_embeddings = self.doc_embeddings.to(self.device)
        with open(doc_id_to_index, 'r') as f:
            self.doc_id_to_index = json.load(f)
        self.embedding_size = self.doc_embeddings.size(1)
            
        
        self.query_embeddings = load(query_embeddings)
        self.query_embeddings = self.query_embeddings.to(self.device)
        
        with open(query_id_to_index, 'r') as f:
            self.query_id_to_index = json.load(f)

        self.aggregation_mode = aggregation_mode
        self.user_model = UserEncoder(aggregation_mode=self.aggregation_mode, embedding_dim=self.embedding_size)

        
        

    def query_encoder(self, queries):
        return self.query_embeddings[tensor([int(self.query_id_to_index[id]) for id in queries])]


    def doc_encoder(self, documents):
        return self.doc_embeddings[tensor([int(self.doc_id_to_index[id]) for id in documents])]
    
            
    
    def forward(self, batch):
        query_embedding = self.query_encoder(batch['query_id'])
        pos_embedding = self.doc_encoder(batch['pos_doc_id'])
        user_doc_embedding = self.doc_encoder(batch['user_doc_id'])

        user_embs = self.user_model(user_doc_embedding, query_embedding)
        return user_embs, pos_embedding, query_embedding
        

    
