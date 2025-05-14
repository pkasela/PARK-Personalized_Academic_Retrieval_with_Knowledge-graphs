import random

from torch import nn, norm, tensor
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

