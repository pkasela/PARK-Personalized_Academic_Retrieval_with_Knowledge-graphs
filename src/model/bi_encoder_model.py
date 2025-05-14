from torch import save, load
from torch import clamp as t_clamp
from torch import nn
from torch import sum as t_sum
from torch import max as t_max
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

class BiEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        max_tokens=512,
        normalize=False,
        pooling_mode='mean',
        device='cpu',
    ):
        super(BiEncoder, self).__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize

        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        self.pooling_mode = pooling_mode

        self.device = device

        self._init_model()


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


    def query_encoder(self, queries):
        encoded_input = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        )
        encoded_input = encoded_input.to(self.device)

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
        )
        encoded_input = encoded_input.to(self.device)
        embeddings = self.doc_model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])

    
    def forward(self, batch):
        query_embedding = self.query_encoder(batch['query'])
        pos_embedding = self.doc_encoder(batch['pos_doc'])
        neg_embedding = self.doc_encoder(batch['neg_doc'])
        
        return {
            'Q_emb': query_embedding,
            'P_emb': pos_embedding,
            'N_emb': neg_embedding
        }
    

    def save(self, path):
        save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(load(path), strict=False)


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
