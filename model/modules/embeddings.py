import math
import torch
import torch.nn as nn
import model.modules.init as my_init
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 dropout=0.0,
                 add_position_embedding=True,
                 padding_idx=0):
        super().__init__()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.padding_idx = padding_idx
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=self.padding_idx)
        self.add_position_embedding = add_position_embedding
        self.scale = embedding_dim ** 0.5
        self.reset_parameters()

    def reset_parameters(self):
        if self.add_position_embedding:
            nn.init.uniform_(self.embeddings.weight, - 1.0 / self.scale, 1.0 / self.scale)
        else:
            my_init.embedding_init(self.embeddings.weight)
        with torch.no_grad():
            self.embeddings.weight[self.padding_idx].fill_(0.0)

    def _add_pos_embedding(self, x, min_timescale=1.0, max_timescale=1.0e4):
        batch, length, channels = list(x.size())
        assert (channels % 2 == 0)
        num_timescales = channels // 2
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (float(num_timescales) - 1.))
        position = torch.arange(0, length).float()
        inv_timescales = torch.arange(0, num_timescales).float()
        if x.is_cuda:
            position = position.cuda()
            inv_timescales = inv_timescales.cuda()

        inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
        scaled_time = position.unsqueeze(1).expand(
            length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
        # scaled time is now length x num_timescales
        # length x channels
        signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)
        return signal.unsqueeze(0).expand(batch, length, channels)

    def forward(self, x):
        ## batch, length, channel
        emb = self.embeddings(x)
        # rescale to [-1.0, 1.0]
        if self.add_position_embedding:
            emb = emb * self.scale
            emb += self._add_pos_embedding(emb)
        if self.dropout is not None:
            emb = self.dropout(emb)
        return emb


class Embedder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_token_embeddings,
                 num_pos_embeddings,
                 num_turn_embeddings,
                 padding_idx=None,
                 dropout=0.1):
        super(Embedder, self).__init__()
        self.token_embeddings = nn.Embedding(num_token_embeddings, hidden_dim,
                                             padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(num_pos_embeddings+1, hidden_dim)
        self.turn_embeddings = nn.Embedding(num_turn_embeddings, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.num_turn_embeddings = num_turn_embeddings
        self.padding_idx = padding_idx
        self._reset_parameters(hidden_dim ** 0.5)

    def _reset_parameters(self, scale):
        nn.init.normal_(self.token_embeddings.weight, mean=0, std=1 / scale)
        nn.init.constant_(self.token_embeddings.weight[self.padding_idx], 0)

    def forward(self, token_inp, turn_inp=None, pos_inp=None):
        input_shape = token_inp.size()  ##[batch, length]
        # print(token_inp)
        # print(turn_ids)
        token_embed = self.token_embeddings(token_inp)
        if turn_inp is None:
            embed = token_embed
        elif len(turn_inp.size()) == 2:
            token_embed += self.turn_embeddings(turn_inp)
            embed = token_embed
        elif turn_inp.size(0) == token_inp.size(0):
            turn_inp = turn_inp.unsqueeze(-1).view(input_shape[0], -1)
            token_embed += self.turn_embeddings(turn_inp)
            embed = token_embed
        else:
            print('trun embedding error')
            
        if pos_inp is not None:
            embed += self.pos_embeddings(pos_inp.long())
            embed = self.drop(embed)
        else:
            pos_inp = torch.arange(token_inp.size(-1), device=token_inp.device, dtype=torch.long)
            embed += self.pos_embeddings(pos_inp) ### for decoder, here do not use dropout 
            
        return embed


if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0], [9, 8, 7, 6, 5, 4, 3, 0, 0]])

    emb = Embedder(hidden_dim=100,
                   num_token_embeddings=50,
                   num_pos_embeddings=20,
                   num_turn_embeddings=3,
                   padding_idx=0)
    out = emb(x, 2)
    print(out)
