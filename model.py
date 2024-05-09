from adapters import init
from torch import save, load
from torch import clamp as t_clamp
from torch import nn, no_grad, tensor
from torch import argmax, arange, stack, einsum, softmax
from torch import sum as t_sum
from torch import max as t_max
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

class GraphBiEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        author_to_index,
        venue_to_index,
        max_tokens=512,
        normalize=False,
        pooling_mode='mean',
        device='cpu',
    ):
        super(GraphBiEncoder, self).__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize

        self.author_to_index = author_to_index
        self.venue_to_index = venue_to_index
        self.num_relations = len(['wrote', 'venue', 'cited'])

        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        self.pooling_mode = pooling_mode

        self.device = device

        self._init_model()
        self.embedding_size = self.doc_model.config.hidden_size

        self.author_embedding = nn.Embedding(
            len(self.author_to_index),
            self.embedding_size,
            device=self.device
        )
        self.venue_embedding = nn.Embedding(
            len(self.venue_to_index),
            self.embedding_size,
            padding_idx=self.venue_to_index[""],
            device=self.device
        )


        self.n_relations = 3
        self.relation_embedding = nn.Embedding(
            self.n_relations,
            self.embedding_size,
            device=self.device
        )

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.author_embedding.weight.data)
        nn.init.xavier_uniform_(self.venue_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)


    def _init_model(self) -> None:
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

        self.doc_model = self.model
        self.q_model = self.model

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if self.pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif self.pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif self.pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif self.pooling_mode == 'identity':
            self.pooling = self.identity


    def add_adapter(self, adapter_name, config):
        init(self.model)
        self.model.add_adapter(adapter_name, config=config, set_active=True)
        self.model.train_adapter(adapter_name) 
        self.model = self.model.to(self.device)


    def query_encoder(self, queries):
        encoded_input = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device)

        embeddings = self.q_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])


    def doc_encoder(self, documents):
        encoded_input = self.tokenizer(
            documents, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device)

        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
    
    
    def cited_embed(self, cited_docs):
        batch_size = len(cited_docs)
        flatten_docs = sum(cited_docs, [])
        cited_embs = self.doc_encoder(flatten_docs)
        cited_embs = cited_embs.view(batch_size, -1, self.embedding_size)
        return cited_embs

    def wrote_embed(self, written_docs):
        batch_size = len(written_docs)
        flatten_docs = sum(written_docs, [])
        wrote_embs = self.doc_encoder(flatten_docs)
        wrote_embs = wrote_embs.view(batch_size, -1, self.embedding_size)
        return wrote_embs

    def venue_embed(self, venue_ids):
        v_ids = tensor([self.venue_to_index[v] for v in venue_ids]).to(self.device)
        venue_embs = self.venue_embedding(v_ids)
        return venue_embs
    

    def user_embeddings(self, user_ids):
        u_ids = tensor([self.author_to_index[u] for u in user_ids]).to(self.device)
        user_embs = self.author_embedding(u_ids)
        return user_embs
            
    
    def forward(self, batch):
        query_embedding = self.query_encoder(batch['query'])
        
        pos_embedding = self.doc_encoder(batch['pos_doc'])
        pos_cited_embeddings = self.cited_embed(batch['pos_cited'])
        pos_venue_embeddings = self.venue_embed(batch['pos_doc_venue'])
        

        neg_embedding = self.doc_encoder(batch['neg_doc'])
        neg_cited_embeddings = self.cited_embed(batch['neg_cited'])
        neg_venue_embeddings = self.venue_embed(batch['neg_doc_venue'])

        user_witten_embs = self.wrote_embed(batch['user_docs'])
        user_embs = self.user_embeddings(batch['user_id'])

        venue_relation = self.relation_embedding(tensor(2).to(self.device))
        wrote_relation = self.relation_embedding(tensor(1).to(self.device))
        cited_relation = self.relation_embedding(tensor(0).to(self.device))

        return {
            'Q_emb': query_embedding,
            'P_emb': pos_embedding,
            'P_cited': pos_cited_embeddings,
            'P_venue': pos_venue_embeddings,
            'N_emb': neg_embedding,
            'N_cited': neg_cited_embeddings,
            'N_venue': neg_venue_embeddings,
            'U_emb': user_embs,
            'U_wrote': user_witten_embs,
            'wrote': wrote_relation.view(1,-1),
            'venue': venue_relation.view(1,-1),
            'cited': cited_relation.view(1,-1)
        }
    

    def save(self, path):
        save(self.state_dict(), path)

    def save_adapter(self, path, adapter_name):
        self.model.save_adapter(path, adapter_name)

    def load(self, path):
        self.load_state_dict(load(path), strict=False)

    def load_adapter(self, path, adapter_name):
        init(self.model)
        self.model.load_adapter(path)
        self.model.set_active_adapters(adapter_name)
        self.model = self.model.to(self.device)

    
    @staticmethod
    def _translation(entity, hyper_plane, device):
        """
        Projects the entity on the hyper plane indicated by the normal vector:
        it can be achieved by calculating the projection of the entity (embedding)
        on the normal vector, and then with a simple vectorial sum we get the projection
        """
        # the hyper plane is normalized following the contraint described in the paper.
        hyper_plane = F.normalize(hyper_plane, p=2, dim=-1).to(device)
        projection_val = t_sum(entity * hyper_plane, dim=-1, keepdims=True).to(device)
        return entity - projection_val * hyper_plane


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


    @staticmethod
    def cls_pooling(model_output, attention_mask):
        last_hidden = model_output["last_hidden_state"]
        # last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden[:, 0]


    @staticmethod
    def identity(model_output, attention_mask):
        return model_output['pooler_output']
    

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]