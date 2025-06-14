from torch import nn, tensor, matmul
from torch import sum as t_sum
from torch.nn import functional as F


from torch import nn, tensor, matmul
from torch.nn import functional as F
from torch import sum as t_sum

class OnlyUserGraphTransX(nn.Module):
    def __init__(
        self,
        n_authors,
        doc_embs,
        n_relations,
        normalize=False,
        mode='transe',
        device='cpu',
    ):
        super(OnlyUserGraphTransX, self).__init__()
        self.n_authors      = n_authors
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
        
        if self.mode == 'transh':
            wrote_embs = self._translation(wrote_embs, self.hyper_plane(tensor(0).to(self.device)), self.device)
            cited_embs = self._translation(cited_embs, self.hyper_plane(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._translation(coauthor_embs, self.hyper_plane(tensor(2).to(self.device)), self.device)
            
        if self.mode == 'transr':
            wrote_embs = self._transform(wrote_embs, self.M(tensor(0).to(self.device)), self.device)
            cited_embs = self._transform(cited_embs, self.M(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._transform(coauthor_embs, self.M(tensor(2).to(self.device)), self.device)
            
        if self.normalize:
            user_embs = F.normalize(user_embs, dim=-1)
            coauthor_embs = F.normalize(wrote_embs, dim=-1)
            

        # 0 wrote (user wrote doc) ok
        # 1 cited (user cited doc) ok
        # 2 co_author with (user co_author user) ok
        wrote_rel = self.relation_embedding(tensor(0).to(self.device)).view(1,-1).expand(wrote_embs.size())
        cited_rel = self.relation_embedding(tensor(1).to(self.device)).view(1,-1).expand(cited_embs.size())
        co_author_rel = self.relation_embedding(tensor(2).to(self.device)).view(1,-1).expand(coauthor_embs.size())
        
        return {
            'user_emb': user_embs,
            'wrote_emb': wrote_embs,
            'cited_emb': cited_embs,
            'coauthor_emb': coauthor_embs,
            'wrote_rel': wrote_rel,
            'cited_rel': cited_rel,
            'co_author_rel': co_author_rel,
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

class UserVenueGraphTransX(nn.Module):
    def __init__(
        self,
        n_authors,
        n_venues,
        doc_embs,
        venue_pad_id,
        n_relations,
        normalize=False,
        mode='transe',
        device='cpu',
    ):
        super(UserVenueGraphTransX, self).__init__()
        self.n_authors      = n_authors
        self.n_venues       = n_venues
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
        if self.mode == 'transh':
            wrote_embs = self._translation(wrote_embs, self.hyper_plane(tensor(0).to(self.device)), self.device)
            cited_embs = self._translation(cited_embs, self.hyper_plane(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._translation(coauthor_embs, self.hyper_plane(tensor(2).to(self.device)), self.device)
            venue_embs = self._translation(venue_embs, self.hyper_plane(tensor(3).to(self.device)), self.device)
        
        if self.mode == 'transr':
            wrote_embs = self._transform(wrote_embs, self.M(tensor(0).to(self.device)), self.device)
            cited_embs = self._transform(cited_embs, self.M(tensor(1).to(self.device)), self.device)
            coauthor_embs = self._transform(coauthor_embs, self.M(tensor(2).to(self.device)), self.device)
            venue_embs = self._transform(venue_embs, self.M(tensor(3).to(self.device)), self.device)
        
        if self.normalize:
            user_embs = F.normalize(user_embs, dim=-1)
            coauthor_embs = F.normalize(wrote_embs, dim=-1)
            venue_embs = F.normalize(venue_embs, dim=-1)
        

        # 0 wrote (user wrote doc) ok
        # 1 cited (user cited doc) ok
        # 2 co_author with (user co_author user) ok
        # 3 in_venue (user in_venue venue) ok
        # 4 affiliated_to (user affiliated_to affiliation) ok
        wrote_rel = self.relation_embedding(tensor(0).to(self.device)).view(1,-1).expand(wrote_embs.size())
        cited_rel = self.relation_embedding(tensor(1).to(self.device)).view(1,-1).expand(cited_embs.size())
        co_author_rel = self.relation_embedding(tensor(2).to(self.device)).view(1,-1).expand(coauthor_embs.size())
        venue_rel = self.relation_embedding(tensor(3).to(self.device)).view(1,-1).expand(venue_embs.size())
        
        return {
            'user_emb': user_embs,
            'wrote_emb': wrote_embs,
            'cited_emb': cited_embs,
            'coauthor_emb': coauthor_embs,
            'venue_emb': venue_embs,
            'wrote_rel': wrote_rel,
            'cited_rel': cited_rel,
            'co_author_rel': co_author_rel,
            'venue_rel': venue_rel
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
