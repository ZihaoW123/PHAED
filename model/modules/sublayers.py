import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_inner (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=False)
        self.relu = nn.GELU() #nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head=None, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()

        if d_head is None:
            assert d_model % n_head == 0
            d_head = d_model // n_head

        self.head_count = n_head

        self.dim_per_head = d_head

        self.model_dim = d_model

        self.linear_keys = nn.Linear(d_model,
                                     n_head * self.dim_per_head)
        self.linear_values = nn.Linear(d_model,
                                       n_head * self.dim_per_head)
        self.linear_query = nn.Linear(d_model,
                                      n_head * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.dim_per_head * n_head, d_model)

    def _split_heads(self, x):

        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2).contiguous()

    def _combine_heads(self, x):

        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None, enc_attn_cache=None, self_attn_cache=None, **kwargs):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key))  # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e4)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
                       .view(batch_size, head_count,
                             query_len, key_len)[:, 0, :, :] \
            .contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]


class MultiHeadedAttentionRelative(MultiHeadedAttention):
    def _compute_relative_attention(self, q, k, v, mask, rel_k, rel_v, dropout):
        """Calculate relative position-aware dot-product self-attention.
        The attention calculation is augmented with learned representations for the
        relative position between each element in q and each element in k and v.
        \alpha = softmax( q(k+rel_k) ); out = \alpha (v+rel_v)
        Args:
            q: a Tensor with shape [batch, heads, qlen, depth].
            k: a Tensor with shape [batch, heads, klen, depth].
            v: a Tensor with shape [batch, heads, klen, depth].
            bias: bias Tensor.
            relative_embedding_keys: a Tensor with shape [(bsz), qlen, klen, depth].
            relative_embedding_values: a Tensor with shape [(bsz), qlen, klen, depth].
            dropout (optional): nn.Dropout.

        Returns:
            Attention weights. [batch, heads, qlen, klen]
            Attention outputs. [batch, heads, qlen, depth]
        """
        QK = torch.einsum("bhqd,bhkd->bhqk", [q, k])
        if rel_k.dim() == 3:
            QR = torch.einsum("bhqd,qkd->bhqk", [q, rel_k])
        elif rel_k.dim() == 4:
            QR = torch.einsum("bhqd,bqkd->bhqk", [q, rel_k])
        logits = QK + QR

        # [bsz, head, qlen, klen]
        if mask is not None:
            logits = logits.masked_fill(mask, -1e4)
        alpha = F.softmax(logits, -1)
        if dropout is not None:
            alpha = dropout(alpha)

        AV = torch.einsum("bhqk,bhkd->bhqd", [alpha, v])
        if rel_v.dim() == 3:
            AR = torch.einsum("bhqk,qkd->bhqd", [alpha, rel_v])
        elif rel_v.dim() == 4:
            AR = torch.einsum("bhqk,bqkd->bhqd", [alpha, rel_v])
        out = AV + AR

        return alpha, out

    def forward(self, key, value, query, mask=None,
                enc_attn_cache=None, self_attn_cache=None,
                rel_attn_kv=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key)) # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=2)
            value_up = torch.cat([value_up_prev, value_up], dim=2)

        query_up = self._split_heads(self.linear_query(query))
        query_up = query_up / math.sqrt(dim_per_head)

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        if mask is not None:
            mask = mask.unsqueeze(1).expand(
                batch_size, head_count, query_len, key_len)

        # do attention
        attn, context = \
            self._compute_relative_attention(
                q=query_up,
                k=key_up,
                v=value_up,
                mask=mask,
                rel_k=rel_attn_kv[0],
                rel_v=rel_attn_kv[1],
                dropout=self.dropout)

        # 3) Apply attention dropout and compute context vectors.
        # context ([batch, length, d_model])
        context = self._combine_heads(context)

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len).mean(1) \
            .contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]