import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.embeddings import Embedder
from model.modules.transformer_xl_utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from model.modules.transformer_xl_utils.log_uniform_sampler import LogUniformSampler, sample_logits
import context_cache as ctx
import copy
from utils import PAD_ID, SOS_ID, EOS_ID


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=True):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(), #nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=True):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# added by yx 20191102
class ContextMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=True):
        super(ContextMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.attn_cache = None

    def forward(self, h, c, attn_mask=None, attn_map=False):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]
        if self.pre_lnorm:
            ##### layer normalization
            h = self.layer_norm(h)

        head_q = self.q_net(h)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)

        if self.attn_cache is None:
            head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
            head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
            head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
            if ctx.IS_INFERRING: self.attn_cache = [head_k, head_v]
        else:
            head_k, head_v = self.attn_cache

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', [head_q, head_k])
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -1e4)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -1e4)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        if attn_map:
            attn_score_tmp = attn_prob.clone()
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', [attn_prob, head_v])
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)
        if attn_map:
            return output, attn_score_tmp
        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=True):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias

        if r.dim() == 2:  # [klen, d]
            # @zzx (2019-11-21): original version
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # klen x n_head x d_head
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head

        elif r.dim() == 3:  # [klen, bsz, d]
            # @zzx (2019-11-21): when pos_emb (r) being fixed wrt padding,
            # it becomes 3-dim, where a batch dim is added.
            r_head_k = r_head_k.view(rlen, bsz, self.n_head, self.d_head)  # klen x n_head x d_head
            BD = torch.einsum('ibnd,jbnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        #            attn_mask = self._rel_shift(attn_mask[:,:,:,None].expand_as(BD))

        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability 
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                #attn_score = attn_score.float().masked_fill(
                #    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                #attn_score = attn_score.float().masked_fill(
                #    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
            elif attn_mask.dim() == 4:  # @zzx: useless, ignore it
                attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        #        print(attn_prob.size(), w_head_v.size())
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.ctx_attn = ContextMultiHeadAttn(n_head, d_model,
                                             d_head, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, enc_out, enc_mask, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        if enc_out is not None:
            output = self.ctx_attn(output, enc_out, attn_mask=enc_mask)

        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout, **kwargs)
        self.ctx_attn = ContextMultiHeadAttn(n_head, d_model,
                                             d_head, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, enc_out, enc_mask, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, attn_map=False):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        if enc_out is not None:
            if attn_map:
                output, attn_maps = self.ctx_attn(output, enc_out, attn_mask=enc_mask, attn_map=attn_map)
            else:
                output = self.ctx_attn(output, enc_out, attn_mask=enc_mask)
        output = self.pos_ff(output)
        if attn_map:
            return output, attn_maps
        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

class MemTransformerLM(nn.Module):
    def __init__(self,
                 embedder,
                 n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=None, pre_lnorm=True,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=None, same_length=False,
                 attn_type=0, clamp_len=-1, ):
        super().__init__()
        if tie_projs is None:
            tie_projs = [False]
        if cutoffs is None:
            cutoffs = []
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        d_head = d_model // n_head if d_head is None else d_head
        self.d_head = d_head

        self.word_emb = embedder # copy.deepcopy(embedder)
        # self.word_emb.token_embeddings.weight = embedder.token_embeddings.weight
        # self.word_emb.turn_embeddings.weight = embedder.turn_embeddings.weight

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.same_length = same_length
        self.clamp_len = clamp_len
        self.final_layer_norm = nn.LayerNorm(d_model)
        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len=None, ext_len=None, mem_len=None, expect_length=None):
        if tgt_len is not None:
            self.tgt_len = tgt_len
        if ext_len is not None:
            self.mem_len = mem_len
        if mem_len is not None:
            self.ext_len = ext_len
        if expect_length is not None:
            self.expect_len = expect_length
            
    def init_mems(self):
        if ctx.ENABLE_CONTEXT:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            # end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            # beg_idx = max(0, end_idx - self.mem_len)
            ctx.tmp_memory_cache.append(hids)
            num = ctx.memory_cache_turn_max_num
            knowlege_length = 0
            if (num+knowlege_length)<len(ctx.tmp_memory_cache) and knowlege_length>0:
                shuffleidx = np.arange(0, knowlege_length-1)
                if ctx.IS_INFERRING == False:
                    np.random.shuffle(shuffleidx)
                tmp = [ctx.tmp_memory_cache[i] for i in shuffleidx]
                new_hids = tmp + [ctx.tmp_memory_cache[knowlege_length-1]] + ctx.tmp_memory_cache[-num:]
                ctx.tmp_memory_cache = tmp + [ctx.tmp_memory_cache[knowlege_length-1]] + ctx.tmp_memory_cache[-num:]
            else:
                new_hids = ctx.tmp_memory_cache[-(num+knowlege_length):]
                ctx.tmp_memory_cache = ctx.tmp_memory_cache[-(num+knowlege_length):]

            for i in range(len(hids)):
                # cat = torch.cat([mems[i], hids[i]], dim=0)
                # new_mems.append(cat[beg_idx:end_idx].detach())
                # new_mems.append(hids[i][:].detach())  # 只保留当前句子的memory情况
                # new_mems.append(hids[i][:])
                hids_i = torch.cat([tmp[i] for tmp in new_hids], dim=0)
                new_mems.append(hids_i[:].detach()) # .detach()
            
            
        return new_mems

    @staticmethod
    def _shift_mem_pos_seq0(self, pos_seq, mem_mask):
        # pos_seq: [mlen+qlen]
        # mem_mask: [mlen, bsz] 1 for padding
        mlen, bsz = mem_mask.size()
        qlen = pos_seq.size(0) - mlen
        pad_num = mem_mask.to(pos_seq).sum(0)  # [bsz]
        # [mlen+qlen, bsz]
        pad_num_seq = torch.cat([pos_seq.new_zeros((mlen, bsz)), pad_num[None, :].expand(qlen, bsz)],
                                dim=0)
        new_pos_seq = pos_seq[:, None] - pad_num_seq
        return new_pos_seq

    @staticmethod
    def _shift_mem_pos_seq(pos_seq, mem_pad_num):
        # pos_seq: [mlen+qlen]
        # mem_mask: [mlen, bsz] 1 for padding
        mlen, bsz = mem_pad_num.size()
        qlen = pos_seq.size(0) - mlen
        # [mlen+qlen, bsz]
        pad_num_seq = torch.cat([mem_pad_num.float(), pos_seq.new_zeros((qlen, bsz))],
                                dim=0)
        new_pos_seq = pos_seq[:, None] - pad_num_seq
        return new_pos_seq

    @staticmethod
    def _shift_pad_pos_seq(pos_seq, mem_pad_num, tgt_mask):
        # pos_seq: [mlen+qlen]
        # mem_mask: [mlen, bsz] 1 for padding
        # tgt_mask: [qlen, bsz] 1 for padding
        mlen, bsz = mem_mask.size()
        qlen = pos_seq.size(0) - mlen

        tgt_pad_num = tgt_mask.to(pos_seq).sum(0)  # [bsz]
        # [mlen+qlen, bsz]
        pad_num_seq = torch.cat([mem_pad_num,
                                 tgt_pad_num[None, :].expand(qlen, bsz)],
                                dim=0)
        new_pos_seq = pos_seq[:, None] - pad_num_seq
        return new_pos_seq
        

    @staticmethod
    def _shift_tgt_pos_seq(self, pos_seq, tgt_mask, bsz):
        # pos_seq: [qlen]
        qlen = pos_seq.size(0)

        pad_num = tgt_mask.to(pos_seq).sum(0)
        if ctx.IS_INFERRING:
            pad_num_seq = (self.tgt_len - self.expect_len)
            new_pos_seq = pos_seq[:, None] - pad_num_seq
            assert (new_pos_seq[0, :]+1 == self.expect_len).all()
            return new_pos_seq

        pad_num = pad_num[None, :].expand(qlen, bsz)
        # [qlen, bsz]
        # tgt_pad_num = tgt_mask.to(pos_seq).sum(0)
        new_pos_seq = pos_seq[:, None] - pad_num
        assert qlen == self.tgt_len
        return new_pos_seq

    @staticmethod
    def _shift_pad_pos_seq_3(self, pos_seq, mem_pad_num, tgt_mask):
        # pos_seq: [mlen+qlen]
        # mem_mask: [mlen, bsz] 1 for padding
        # tgt_mask: [qlen, bsz] 1 for padding
        mlen, bsz = mem_pad_num.size()
        qlen = pos_seq.size(0) - mlen

        if ctx.IS_INFERRING:
            pad_num_seq = torch.cat([mem_pad_num,
                                    pos_seq.new_zeros((qlen, bsz))],
                                    dim=0) + (self.tgt_len - self.expect_len)
            new_pos_seq = pos_seq[:, None] - pad_num_seq
            assert (new_pos_seq[mlen, :]+1 == self.expect_len).all()
            return new_pos_seq

        tgt_pad_num = tgt_mask.to(pos_seq).sum(0)  # [bsz]
        # pos_seq.new_zeros((qlen, bsz))
        # tgt_pad_num[None, :].expand(qlen, bsz)

        pad_num_seq = torch.cat([mem_pad_num+tgt_pad_num[None, :],
                                tgt_pad_num[None, :].expand(qlen, bsz)],
                                dim=0)
        # [mlen+qlen, bsz]
        new_pos_seq = pos_seq[:, None] - pad_num_seq
        assert mlen % qlen == 0
        return new_pos_seq

    def _forward(self, dec_inp, dec_turn, enc_out, enc_mask, mems=None, attn_map=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp.transpose(0, 1), turn_inp=dec_turn, pos_inp=None).transpose(0, 1)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                             + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None].bool()  # -1
        else:
            # [qlen, mlen+qlen, 1]
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen).long()[:, :, None].bool()
            if ctx.ENABLE_CONTEXT and ctx.memory_mask is not None:
                mem_mask = ctx.memory_mask[-mlen:]  # [mlen, bsz]
                new_mem_mask = torch.cat(
                    [mem_mask, mem_mask.new_full((qlen, bsz), 0)],
                    dim=0)  # [mlen+qlen, bsz]
                dec_attn_mask = (dec_attn_mask + new_mem_mask[None, :, :]).gt(0)
            

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            #if ctx.IS_INFERRING:
            #    pos_seq = torch.arange(mlen+self.tgt_len-1, -1, -1.0, device=word_emb.device,
            #                       dtype=word_emb.dtype)
            #    pos_seq = pos_seq[:klen]

            ##
 
            # dealing with Padding problem within position embeddings
            if ctx.ENABLE_CONTEXT and ctx.memory_mask is not None:
                mem_mask = ctx.memory_mask[-mlen:]  # [mlen, bsz]
                mem_pad_num = ctx.memory_pad_num[-mlen:]
                tgt_mask = torch.gt((dec_inp.eq(PAD_ID) + dec_inp.eq(EOS_ID)), 0)  # [qlen bsz]
                # [klen, bsz]
                # pos_seq = self._shift_pad_pos_seq_3(self, pos_seq, mem_pad_num.float(), tgt_mask)
                # pos_seq = self._shift_pad_pos_seq(pos_seq, mem_pad_num, tgt_mask)
                pos_seq = self._shift_mem_pos_seq(pos_seq, mem_pad_num)

            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len, min=0)

            if ctx.ENABLE_CONTEXT and ctx.memory_mask is not None:
                pos_emb = self.pos_emb(pos_seq.view(-1)).view(klen, bsz, -1)  # [klen, bsz, d]
            else:
                pos_emb = self.pos_emb(pos_seq)  # [klen, 1, d]
                pos_emb = pos_emb.squeeze(1)  # [klen, d] batch==1

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if attn_map:
                    core_out, attn_maps = layer(core_out, pos_emb, enc_out, enc_mask, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i, attn_map=attn_map)
                else:
                    core_out = layer(core_out, pos_emb, enc_out, enc_mask, self.r_w_bias,
                                 self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)

                hids.append(core_out)

        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        
        core_out = self.final_layer_norm(core_out)
        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)
        if attn_map:
            return core_out, new_mems, attn_maps
        return core_out, new_mems

    def forward(self, dec_inp, dec_turn, enc_out, enc_mask, *mems, attn_map=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: 
            mems = self.init_mems()
            ctx.tmp_memory_cache = []

        if attn_map:
            output, new_mems, attn_maps = self._forward(dec_inp, dec_turn, enc_out, enc_mask, mems=mems, attn_map=attn_map)
            return  output, new_mems, attn_maps
            
        output, new_mems = self._forward(dec_inp, dec_turn, enc_out, enc_mask, mems=mems)
        if not ctx.ENABLE_CONTEXT:
            assert new_mems is None and mems is None
        return output, new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                                     args.d_model, args.d_head, args.d_inner, args.dropout,
                                     dropatt=args.dropout, tie_weight=True,
                                     d_embed=d_embed, div_val=div_val,
                                     tie_projs=tie_projs, pre_lnorm=True,
                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
