from torch import concat, einsum, nn, mm, stack, tensor, Tensor
from torch.nn.functional import relu, log_softmax, softmax

class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, output) -> Tensor:
        pos_scores = einsum('ij,ij->i', output['Q_emb'], output['P_emb']) #output['Q_emb'] @ output['P_emb'].T
        neg_scores = einsum('ij,ij->i', output['Q_emb'], output['N_emb']) #output['Q_emb'] @ output['N_emb'].T
        hard_loss = relu(self.margin - pos_scores + neg_scores).mean()
        accuracy = pos_scores > neg_scores

        # only you can have a batch with more than 1 element do random negative batching:
        
        if output['P_emb'].shape[0] > 1:
            repeated_pos_scores = pos_scores.view(-1,1).repeat_interleave(
                (output['P_emb'].shape[0] - 1)*2, dim=1
            )
            query_pos_scores = output['Q_emb'] @ output['P_emb'].T # batch_size * batch_size
            
            masks = [[j for j in range(output['Q_emb'].shape[0]) if j != i] for i in range(output['Q_emb'].shape[0])]
            neg_docs_score_1 = query_pos_scores.gather(1, tensor(masks).to(query_pos_scores.device))

            query_neg_scores = output['Q_emb'] @ output['N_emb'].T # batch_size * batch_size
            neg_docs_score_2 = query_neg_scores.gather(1, tensor(masks).to(query_neg_scores.device))

            neg_docs_score = concat((neg_docs_score_1, neg_docs_score_2), dim=1)
            
            in_batch_loss = relu(self.margin - repeated_pos_scores + neg_docs_score).mean()
            hard_loss = hard_loss + in_batch_loss
        
        
        
        return hard_loss, accuracy
    