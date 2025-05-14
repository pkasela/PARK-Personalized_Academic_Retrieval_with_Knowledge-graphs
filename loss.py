import random

from torch import concat, einsum, nn, norm, mm, stack, tensor, Tensor
from torch.nn.functional import relu


def shuffle_list(some_list):
    randomized_list = some_list[:]
    while True:
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
        else:
            return randomized_list

class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, output) -> Tensor:
        ranking_loss, accuracy = self.ranking_loss(output)
        graph_loss = self.graph_loss(output)

        # pos_scores = einsum('ij,ij->i', output['U_emb'] + output['cited'].expand(output['U_emb'].size()), output['P_emb']) 
        # neg_scores = einsum('ij,ij->i', output['U_emb'] + output['cited'].expand(output['U_emb'].size()), output['N_emb']) 
        
        # accuracy = pos_scores > neg_scores
        return 0.9*ranking_loss + 0.1*graph_loss, accuracy
        
    def graph_loss(self, output) -> Tensor:
        batch_size = output['U_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['U_emb'].device)
        # user wrote paper:
        U_emb = output['U_emb']
        wrote_emb = output['wrote']
        wrote_emb = wrote_emb.expand(U_emb.size())
        U_wrote_embs = output['U_wrote']

        
        U_emb = U_emb.unsqueeze(dim=1).expand(U_wrote_embs.size())
        wrote_emb = wrote_emb.unsqueeze(dim=1).expand(U_wrote_embs.size())

        pos_user_score = relu(norm(U_emb + wrote_emb - U_wrote_embs, p=2, dim=-1))
        neg_user_score = relu(norm(U_emb + wrote_emb - U_wrote_embs[negative_index], p=2, dim=-1))

        user_wrote_score = relu(self.margin - pos_user_score + neg_user_score).mean()

        """P_emb = output['P_emb']
        pos_venue = output['P_venue']
        venue_emb = output['venue'].expand(pos_venue.size())

        pos_venue_score = relu(norm(P_emb + venue_emb - pos_venue, p=2, dim=-1))
        neg_venue_score = relu(norm(P_emb + venue_emb - pos_venue[negative_index], p=2, dim=-1))

        p_venue_score = relu(self.margin - pos_venue_score + neg_venue_score).mean()

        N_emb = output['N_emb']
        neg_venue = output['N_venue']
        venue_emb = output['venue'].expand(neg_venue.size())

        pos_venue_score = relu(norm(N_emb + venue_emb - neg_venue, p=2, dim=-1))
        neg_venue_score = relu(norm(N_emb + venue_emb - neg_venue[negative_index], p=2, dim=-1))
        n_venue_score = relu(self.margin - pos_venue_score + neg_venue_score).mean()
        
        P_emb = output['P_emb']
        P_cited_embs = output['P_cited']
        cited_emb = output['cited']
        cited_emb = cited_emb.expand(P_emb.size())

        
        P_emb = P_emb.unsqueeze(dim=1).expand(P_cited_embs.size())
        cited_emb = cited_emb.unsqueeze(dim=1).expand(P_cited_embs.size())

        pos_cited_score = relu(norm(P_emb + cited_emb - P_cited_embs, p=2, dim=-1))
        neg_cited_score = relu(norm(P_emb + cited_emb - P_cited_embs[negative_index], p=2, dim=-1))

        p_cited_score = relu(self.margin - pos_cited_score + neg_cited_score).mean()


        N_emb = output['N_emb']
        N_cited_embs = output['N_cited']
        
        N_emb = N_emb.unsqueeze(dim=1).expand(N_cited_embs.size())

        pos_cited_score = relu(norm(N_emb + cited_emb - N_cited_embs, p=2, dim=-1))
        neg_cited_score = relu(norm(N_emb + cited_emb - N_cited_embs[negative_index], p=2, dim=-1))

        n_cited_score = relu(self.margin - pos_cited_score + neg_cited_score).mean()
        
        U_emb = output['U_emb']
        P_emb = output['P_emb']
        N_emb = output['N_emb']
        cited_emb = output['cited']
        cited_emb = cited_emb.expand(P_emb.size())
        
        
        pos_u_cited_score = relu(norm(U_emb + cited_emb - P_emb, p=2, dim=-1))
        # neg_u_cited_score = relu(norm(U_emb + cited_emb - P_emb[negative_index], p=2, dim=-1))
        neg_u_cited_score_2 = relu(norm(U_emb + cited_emb - N_emb, p=2, dim=-1))

        # u_cited_score = relu(self.margin - pos_u_cited_score + neg_u_cited_score).mean()
        u_cited_score_2 = relu(self.margin - pos_u_cited_score + neg_u_cited_score_2).mean()
        """
        
        graph_loss = user_wrote_score
        return graph_loss

    def graph_loss_transh(self, output) -> Tensor:
        batch_size = output['U_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['U_emb'].device)
        # user wrote paper:
        U_emb_wrote_head = output['U_wrote_head']
        wrote_emb = output['wrote']
        wrote_emb = wrote_emb.expand(U_emb_wrote_head.size())
        U_wrote_embs = output['U_wrote_head']

        pos_user_score = relu(norm(U_emb_wrote_head + wrote_emb - U_wrote_embs, p=2, dim=-1))
        neg_user_score = relu(norm(U_emb_wrote_head + wrote_emb - U_wrote_embs[negative_index], p=2, dim=-1))

        user_wrote_score = relu(self.margin - pos_user_score + neg_user_score).mean()
        
        U_emb_cited = output['U_cited_head']
        P_emb_cited = output['P_emb_cited_tail']
        N_emb_cited = output['N_emb_cited_tail']
        cited_emb = output['cited']
        cited_emb = cited_emb.expand(P_emb_cited.size())
        
        
        pos_u_cited_score = relu(norm(U_emb_cited + cited_emb - P_emb_cited, p=2, dim=-1))
        neg_u_cited_score = relu(norm(U_emb_cited + cited_emb - N_emb_cited, p=2, dim=-1))

        u_cited_score = relu(self.margin - pos_u_cited_score + neg_u_cited_score).mean()
        
        
        graph_loss = user_wrote_score + u_cited_score
        return graph_loss
        

    def ranking_loss(self, output) -> Tensor:

        #ranking loss 
        # cited_emb = output['cited']
        #  = cited_emb.expand(output['U_emb'].size())
        
        pos_scores = einsum('ij,ij->i', output['U_emb'], output['P_emb']) #output['Q_emb'] @ output['P_emb'].T
        neg_scores = einsum('ij,ij->i', output['U_emb'], output['N_emb']) #output['Q_emb'] @ output['N_emb'].T
        hard_loss = relu(self.margin - pos_scores + neg_scores).mean()
        accuracy = pos_scores > neg_scores

        # only you can have a batch with more than 1 element do random negative batching:
        if output['P_emb'].shape[0] > 1:
            repeated_pos_scores = pos_scores.view(-1,1).repeat_interleave(
                (output['P_emb'].shape[0] - 1)*2, dim=1
            )
            query_pos_scores = (output['U_emb']) @ output['P_emb'].T # batch_size * batch_size
            
            masks = [[j for j in range(output['U_emb'].shape[0]) if j != i] for i in range(output['U_emb'].shape[0])]
            neg_docs_score_1 = query_pos_scores.gather(1, tensor(masks).to(query_pos_scores.device))

            query_neg_scores = (output['U_emb']) @ output['N_emb'].T # batch_size * batch_size
            neg_docs_score_2 = query_neg_scores.gather(1, tensor(masks).to(query_neg_scores.device))

            neg_docs_score = concat((neg_docs_score_1, neg_docs_score_2), dim=1)
            
            in_batch_loss = relu(self.margin - repeated_pos_scores + neg_docs_score).mean()
            hard_loss = hard_loss + in_batch_loss

        
        
        return hard_loss, accuracy
    

class TransXLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TransXLoss, self).__init__()
        self.margin = margin

    def forward(self, output):
        batch_size = output['user_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['user_emb'].device)
        # user wrote doc
        pos_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'][negative_index], p=2, dim=-1)
        wrote_score = relu(self.margin + pos_score - neg_score).mean()

        # user cited doc
        pos_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'][negative_index], p=2, dim=-1)
        cited_score = relu(self.margin + pos_score - neg_score).mean()

        # user coauthor user
        pos_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'][negative_index], p=2, dim=-1)
        coauthor_score = relu(self.margin + pos_score - neg_score).mean()

        # user venue venue
        pos_score = norm(output['user_emb'] + output['venue_rel'] - output['venue_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['venue_rel'] - output['venue_emb'][negative_index], p=2, dim=-1)
        venue_score = relu(self.margin + pos_score - neg_score).mean()
        
        # user venue venue
        pos_score = norm(output['user_emb'] + output['affiliation_rel'] - output['affiliation_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['affiliation_rel'] - output['affiliation_emb'][negative_index], p=2, dim=-1)
        affiliation_score = relu(self.margin + pos_score - neg_score).mean()

        return wrote_score + cited_score + coauthor_score + venue_score + affiliation_score



class OnlyUserTransXLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(OnlyUserTransXLoss, self).__init__()
        self.margin = margin

    def forward(self, output):
        batch_size = output['user_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['user_emb'].device)
        # user wrote doc
        pos_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'][negative_index], p=2, dim=-1)
        wrote_score = relu(self.margin + pos_score - neg_score).mean()

        # user cited doc
        pos_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'][negative_index], p=2, dim=-1)
        cited_score = relu(self.margin + pos_score - neg_score).mean()

        # user coauthor user
        pos_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'][negative_index], p=2, dim=-1)
        coauthor_score = relu(self.margin + pos_score - neg_score).mean()

        return wrote_score + cited_score + coauthor_score

class UserAffilTransXLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(UserAffilTransXLoss, self).__init__()
        self.margin = margin

    def forward(self, output):
        batch_size = output['user_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['user_emb'].device)
        # user wrote doc
        pos_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'][negative_index], p=2, dim=-1)
        wrote_score = relu(self.margin + pos_score - neg_score).mean()

        # user cited doc
        pos_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'][negative_index], p=2, dim=-1)
        cited_score = relu(self.margin + pos_score - neg_score).mean()

        # user coauthor user
        pos_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'][negative_index], p=2, dim=-1)
        coauthor_score = relu(self.margin + pos_score - neg_score).mean()

        
        # user affil affil
        pos_score = norm(output['user_emb'] + output['affiliation_rel'] - output['affiliation_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['affiliation_rel'] - output['affiliation_emb'][negative_index], p=2, dim=-1)
        affiliation_score = relu(self.margin + pos_score - neg_score).mean()

        return wrote_score + cited_score + coauthor_score + affiliation_score


class UserVenueTransXLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(UserVenueTransXLoss, self).__init__()
        self.margin = margin

    def forward(self, output):
        batch_size = output['user_emb'].size()[0]
        negative_index = tensor(shuffle_list(list(range(batch_size)))).to(output['user_emb'].device)
        # user wrote doc
        pos_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['wrote_rel'] - output['wrote_emb'][negative_index], p=2, dim=-1)
        wrote_score = relu(self.margin + pos_score - neg_score).mean()

        # user cited doc
        pos_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['cited_rel'] - output['cited_emb'][negative_index], p=2, dim=-1)
        cited_score = relu(self.margin + pos_score - neg_score).mean()

        # user coauthor user
        pos_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['co_author_rel'] - output['coauthor_emb'][negative_index], p=2, dim=-1)
        coauthor_score = relu(self.margin + pos_score - neg_score).mean()

        # user venue venue
        pos_score = norm(output['user_emb'] + output['venue_rel'] - output['venue_emb'], p=2, dim=-1)
        neg_score = norm(output['user_emb'] + output['venue_rel'] - output['venue_emb'][negative_index], p=2, dim=-1)
        venue_score = relu(self.margin + pos_score - neg_score).mean()
        
        return wrote_score + cited_score + coauthor_score + venue_score

