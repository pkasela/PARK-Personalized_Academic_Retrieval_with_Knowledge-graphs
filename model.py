from torch import save, load, concat
from torch import clamp as t_clamp
from torch import nn, no_grad, tensor, matmul
from torch import sum as t_sum
from torch import max as t_max
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import json

class GraphBiEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        author_to_index,
        venue_to_index,
        doc_embeddings=None,
        doc_id_to_index=None,
        query_embeddings=None,
        query_id_to_index=None,
        max_tokens=512,
        normalize=True,
        pooling_mode='mean',
        device='cpu',
    ):
        super(GraphBiEncoder, self).__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.embedding_mode = 'model'
        
        self.author_to_index = author_to_index
        self.venue_to_index = venue_to_index
        self.num_relations = len(['wrote', 'venue', 'cited'])

        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        self.pooling_mode = pooling_mode

        self.device = device

        self._init_model()
        self.embedding_size = self.doc_model.config.hidden_size
        if doc_embeddings:
            self.doc_embeddings = load(doc_embeddings)
            self.doc_embeddings = self.doc_embeddings.to(self.device)
            with open(doc_id_to_index, 'r') as f:
                self.doc_id_to_index = json.load(f)
                self.doc_id_to_index['-1'] = len(self.doc_id_to_index) # for [SEP] token
            with no_grad():
                padding_vector = self.doc_encoder(['[SEP]'])
            self.embedding_mode = 'pretrained'
            self.doc_embeddings = concat((self.doc_embeddings, padding_vector), dim=0)

        if query_embeddings:
            self.query_embeddings = load(query_embeddings)
            with open(query_id_to_index, 'r') as f:
                self.query_id_to_index = json.load(f)


        self.author_embedding = nn.Embedding(
            len(self.author_to_index) + 1,
            self.embedding_size,
            padding_idx=len(self.author_to_index),
            device=self.device
        )
        self.venue_embedding = nn.Embedding(
            len(self.venue_to_index),
            self.embedding_size,
            padding_idx=self.venue_to_index[""],
            device=self.device
        )


        self.n_relations = 2
        self.relation_embedding = nn.Embedding(
            self.n_relations,
            self.embedding_size,
            device=self.device
        )
        self.hyper_plane = nn.Embedding(
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
        if self.embedding_mode == 'model':
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
    
        elif self.embedding_mode == 'pretrained':
            return self.query_embeddings[tensor([int(self.query_id_to_index[id]) for id in queries])]


    def doc_encoder(self, documents):
        if self.embedding_mode == 'model':
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
        
        elif self.embedding_mode == 'pretrained':
            return self.doc_embeddings[tensor([int(self.doc_id_to_index[id]) for id in documents])]
    
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
        wrote_hyperplane = self.hyper_plane(tensor(1).to(self.device))
        wrote_embs = self._translation(wrote_embs, wrote_hyperplane, self.device)
        if self.normalize:
            wrote_embs = F.normalize(wrote_embs, dim=-1)
        wrote_embs = wrote_embs.view(batch_size, -1, self.embedding_size)
        return wrote_embs, wrote_hyperplane

    def venue_embed(self, venue_ids):
        v_ids = tensor([self.venue_to_index[v] for v in venue_ids]).to(self.device)
        venue_embs = self.venue_embedding(v_ids)
        return venue_embs
    

    def user_embeddings(self, user_ids):
        u_ids = tensor([self.author_to_index.get(u, len(self.author_to_index)) for u in user_ids]).to(self.device)
        user_embs = self.author_embedding(u_ids)
        if self.normalize:
            return F.normalize(user_embs, dim=-1)
            
        return user_embs
            
    
    def forward(self, batch):
        with no_grad():
            query_embedding = self.query_encoder(batch['query_id'])
        
            pos_embedding = self.doc_encoder(batch['pos_doc_id'])
            # pos_cited_embeddings = self.cited_embed(batch['pos_cited_id'])
        
        # pos_venue_embeddings = self.venue_embed(batch['pos_doc_venue'])
        
        with no_grad():
            neg_embedding = self.doc_encoder(batch['neg_doc_id'])
            # neg_cited_embeddings = self.cited_embed(batch['neg_cited_id'])
        # neg_venue_embeddings = self.venue_embed(batch['neg_doc_venue'])

        with no_grad():
            user_witten_embs, wrote_hyperplane = self.wrote_embed(batch['user_docs_id'])
        user_embs = self.user_embeddings(batch['user_id'])

        # venue_relation = self.relation_embedding(tensor(2).to(self.device))
        wrote_relation = self.relation_embedding(tensor(1).to(self.device))
        cited_relation = self.relation_embedding(tensor(0).to(self.device))

        u_wrote_head = self._translation(user_embs, wrote_hyperplane, self.device)
        if self.normalize:
            u_wrote_head = F.normalize(u_wrote_head, dim=-1)

        cited_hyperplane = self.hyper_plane(tensor(0).to(self.device))

        u_cited_head = self._translation(user_embs, cited_hyperplane, self.device)
        pos_embedding_tail = self._translation(pos_embedding, cited_hyperplane, self.device)
        if self.normalize:
            pos_embedding_tail = F.normalize(pos_embedding_tail, dim=-1)
        neg_embedding_tail = self._translation(neg_embedding, cited_hyperplane, self.device)
        if self.normalize:
            neg_embedding_tail = F.normalize(neg_embedding_tail, dim=-1)

        return {
            'Q_emb': query_embedding,
            'P_emb': pos_embedding,
            'P_emb_cited_tail': pos_embedding_tail,
            # 'P_cited': pos_cited_embeddings,
            # 'P_venue': pos_venue_embeddings,
            'N_emb': neg_embedding,
            'N_emb_cited_tail': neg_embedding_tail,
            # 'N_cited': neg_cited_embeddings,
            # 'N_venue': neg_venue_embeddings,
            'U_emb': user_embs,
            'U_wrote_head': u_wrote_head,
            'U_wrote': user_witten_embs,
            'U_cited_head': u_cited_head,
            'wrote': wrote_relation.view(1,-1),
            # 'venue': venue_relation.view(1,-1),
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
    



class GraphTransH(nn.Module):
    def __init__(
        self,
        n_authors,
        n_venues,
        n_affiliations,
        doc_embs,
        venue_pad_id,
        affiliation_pad_id,
        n_relations,
        normalize=False,
        mode='transe',
        device='cpu',
    ):
        super(GraphTransH, self).__init__()
        self.n_authors      = n_authors
        self.n_venues       = n_venues
        self.n_affiliations = n_affiliations
        self.device         = device
        self.mode           = mode
        self.normalize      = normalize
        
        self.doc_embedding  = doc_embs
        self.embedding_size = self.doc_embedding.shape[1]
        self.author_embedding = nn.Embedding(
            n_authors,
            self.embedding_size,
            device=self.device
        )
        self.venue_embedding = nn.Embedding(
            n_venues,
            self.embedding_size,
            padding_idx=venue_pad_id,
            device=self.device
        )
        self.affliation_embedding = nn.Embedding(
            n_affiliations,
            self.embedding_size,
            padding_idx=affiliation_pad_id,
            device=self.device
        )

        self.n_relations = n_relations
        self.relation_embedding = nn.Embedding(
            self.n_relations,
            self.embedding_size,
            device=self.device
        )
        if self.mode == 'transh':
            self.hyper_plane = nn.Embedding(
                self.n_relations,
                self.embedding_size,
                device=self.device
            )
        if self.mode == 'transr':
            self.M = nn.Embedding(
                self.n_relations,
                self.embedding_size*self.embedding_size,
                device=self.device
            )

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.author_embedding.weight.data)
        nn.init.xavier_uniform_(self.venue_embedding.weight.data)
        nn.init.xavier_uniform_(self.affliation_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        if self.mode == 'transh':
            nn.init.xavier_uniform_(self.hyper_plane.weight.data)
        if self.mode == 'transr':
            nn.init.xavier_uniform_(self.M.weight.data)
        
    def forward(self, data):

        user_embs = self.author_embedding(tensor(data['user_id']).to(self.device))

        wrote_embs = self.doc_embedding[tensor(data['wrote']).to(self.device)]
        cited_embs = self.doc_embedding[tensor(data['cited']).to(self.device)]
        coauthor_embs = self.author_embedding(tensor(data['coauthor']).to(self.device))
        venue_embs = self.venue_embedding(tensor(data['venue']).to(self.device))
        affiliation_embs = self.affliation_embedding(tensor(data['affiliation']).to(self.device))
        if self.mode == 'transh':
            wrote_embs = self._translation(wrote_embs, self.hyper_plane(tensor(0).to(self.device)), self.device)
            cited_embs = self._translation(cited_embs, self.hyper_plane(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._translation(coauthor_embs, self.hyper_plane(tensor(2).to(self.device)), self.device)
            venue_embs = self._translation(venue_embs, self.hyper_plane(tensor(3).to(self.device)), self.device)
            affiliation_embs = self._translation(affiliation_embs, self.hyper_plane(tensor(4).to(self.device)), self.device)

        if self.mode == 'transr':
            wrote_embs = self._transform(wrote_embs, self.M(tensor(0).to(self.device)), self.device)
            cited_embs = self._transform(cited_embs, self.M(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._transform(coauthor_embs, self.M(tensor(2).to(self.device)), self.device)
            venue_embs = self._transform(venue_embs, self.M(tensor(3).to(self.device)), self.device)
            affiliation_embs = self._transform(affiliation_embs, self.M(tensor(4).to(self.device)), self.device)

        if self.normalize:
            user_embs = F.normalize(user_embs, dim=-1)
            coauthor_embs = F.normalize(wrote_embs, dim=-1)
            venue_embs = F.normalize(venue_embs, dim=-1)
            affiliation_embs = F.normalize(affiliation_embs, dim=-1)
        

        # 0 wrote (user wrote doc) ok
        # 1 cited (user cited doc) ok
        # 2 co_author with (user co_author user) ok
        # 3 in_venue (user in_venue venue) ok
        # 4 affiliated_to (user affiliated_to affiliation) ok
        wrote_rel = self.relation_embedding(tensor(0).to(self.device)).view(1,-1).expand(wrote_embs.size())
        cited_rel = self.relation_embedding(tensor(1).to(self.device)).view(1,-1).expand(cited_embs.size())
        co_author_rel = self.relation_embedding(tensor(2).to(self.device)).view(1,-1).expand(coauthor_embs.size())
        venue_rel = self.relation_embedding(tensor(3).to(self.device)).view(1,-1).expand(venue_embs.size())
        affiliation_rel = self.relation_embedding(tensor(4).to(self.device)).view(1,-1).expand(affiliation_embs.size())

        return {
            'user_emb': user_embs,
            'wrote_emb': wrote_embs,
            'cited_emb': cited_embs,
            'coauthor_emb': coauthor_embs,
            'venue_emb': venue_embs,
            'affiliation_emb': affiliation_embs,
            'wrote_rel': wrote_rel,
            'cited_rel': cited_rel,
            'co_author_rel': co_author_rel,
            'venue_rel': venue_rel,
            'affiliation_rel': affiliation_rel
        }
    
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
    def _transform(v, M, device):
        """
        Projects the entities v in another space using M for each relation r
        torch.matmul uses batch matrix product if n_dimension > 2
        The first dimension indicates the batch while the multiplication
        is done on the 2 and 3 dimension in this case thus:
        M x v -> (batch, rel_dim, entity_dim) x (entity_dim, 1) will return
        (batch, rel_dim, 1) or the entity vector in the relation space
        """
        entity_dim = v.shape[1]
        rel_dim = M.shape[0]//entity_dim
        v = v.view(-1, entity_dim, 1).to(device)
        M = M.view(-1, rel_dim, entity_dim).to(device)
        return matmul(M, v).view(-1, rel_dim).to(device)