import torch
import torch.nn as nn
import numpy as np

# PE

# for changeable length
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(PositionalEncoding, self).__init__()
#         # PE matrix
#         position_encoding = np.array([
#             [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
#             for pos in range(max_seq_len)
#         ])
#         position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
#         position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
#         position_encoding = torch.from_numpy(position_encoding)
#
#         pad_row = torch.zeros([1, d_model])
#         position_encoding = torch.cat((pad_row, position_encoding))
#         self.position_encoding = nn.Embedding(max_seq_len+1, d_model)
#         self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
#
#     def forward(self, input_len):
#         max_len = torch.max(input_len)
#         input_pos = torch.tensor(
#             [list(range(1, len_idx+1)) + [0] * (max_len - len_idx) for len_idx in input_len]
#         )
#         if input_len.is_cuda:
#             input_pos = input_pos.cuda()
#         return self.position_encoding(input_pos)

# for fixed length
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, requires_grad=False):
        super(PositionalEncoding, self).__init__()
        position_encoding = np.array([
            [pos / np.power(10000, 2.0*(j//2) / d_model) for j in range(d_model)]
            for pos in range(seq_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding)
        self.position_encoding = nn.Parameter(position_encoding, requires_grad=requires_grad)

    def forward(self):
        return self.position_encoding

# Attention
class ScaleDotProductAttention(nn.Module):
    """
        Scaled dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        forward
        :param q: Queries tensor, [B, num_heads, L_q, D_q]
        :param k: Keys tensor, [B, num_heads, L_k, D_k]
        :param v: Values tensor, [B, num_heads, L_v, D_v]
        :param scale: scale factor
        :param attn_mask: Masking tensor, [B, num_heads, L_q, L_k]
        :return:context tensor, attention tensor
        """
        attention = torch.matmul(q, k.transpose(-1, -2))
        if scale:
            attention *= scale
        if attn_mask != None:
            attention = attention.masked_fill(attn_mask, float('-inf'))
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    """
        Multi-head attention mechanism
    """
    def __init__(self, model_dim=512, out_dim=512, num_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.out_dim = out_dim

        self.linear_k = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head*num_heads)

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, key, value, query, attn_mask=None):
        # residual connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)

        if attn_mask != None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # concat heads
        context = context.transpose(1, 2).reshape(batch_size, -1, dim_per_head*num_heads)

        # linear
        output = self.linear_final(context)
        output = self.dropout(output)

        # add residual and norm layer
        if self.model_dim == self.out_dim:
            output += residual
        output = self.layer_norm(output)

        return output, attention

# Feed-forward
class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(nn.ReLU(inplace=True)((self.w1(output))))
        output = self.dropout(output.transpose(1, 2))
        output += x
        output = self.layer_norm(output)
        return output


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, out_dim=512, num_heads=8, ffn_dim=2048, dropout=0.):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, out_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(out_dim, ffn_dim, dropout)
    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention



























