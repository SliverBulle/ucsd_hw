
import torch
import torch.nn as nn
import math

# feedforward classifier
class FeedforwardClassifier(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class AliBi_decoder(nn.Module):
    def __init__(self, max_seq_len, n_head, device):
        super(AliBi_decoder, self).__init__()
        self.bias = self._get_slopes(n_head, max_seq_len).to(device)

    def _get_slopes(self, n_head, max_seq_len):
        slopes = [1 / (2 ** i) for i in range(1, n_head + 1)]
        slopes = torch.tensor(slopes).float().unsqueeze(1).unsqueeze(2)
        positions = torch.arange(max_seq_len).float()
        diff = positions.unsqueeze(0).unsqueeze(1) - positions.unsqueeze(0).unsqueeze(2)
        # 应用因果掩码
        causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len)))
        diff = diff * causal_mask
        bias = slopes * diff
        return bias.unsqueeze(0)

    def forward(self, query_length, key_length):
        # 确保返回的偏置也遵循因果掩码
        bias = self.bias[:, :, :query_length, :key_length]
        # 将非因果位置的偏置设为负无穷
        causal_mask = torch.tril(torch.ones((query_length, key_length), device=bias.device))
        bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        return bias
# AliBi position encoding
class AliBi(nn.Module):
    def __init__(self, max_seq_len, n_head, device):
        super(AliBi, self).__init__()
        self.bias = self._get_slopes(n_head, max_seq_len).to(device)

    def _get_slopes(self, n_head, max_seq_len):
        """
        生成 AliBi 偏置矩阵，形状为 (1, n_head, max_seq_len, max_seq_len)
        """
        slopes = [1 / (2 ** i) for i in range(1, n_head + 1)]
        slopes = torch.tensor(slopes).float().unsqueeze(1).unsqueeze(2)  # 形状: (n_head, 1, 1)
        positions = torch.arange(max_seq_len).float()  # (max_seq_len,)
        # 计算查询位置和键位置之间的相对距离
        diff = positions.unsqueeze(0).unsqueeze(1) - positions.unsqueeze(0).unsqueeze(2)  # (1, max_seq_len, max_seq_len)
        bias = slopes * diff  # (n_head, max_seq_len, max_seq_len)
        return bias.unsqueeze(0)  # (1, n_head, max_seq_len, max_seq_len)

    def forward(self, query_length, key_length):
        return self.bias[:, :, :key_length, :key_length]  # 确保返回的偏置形状为 (1, n_head, key_length, key_length)

# standard multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_alibi=False, max_seq_len=512):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_alibi = use_alibi

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        if self.use_alibi:
            self.alibi = AliBi(max_seq_len, n_head, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x, encoder_out=None):
        B, T, C = x.size()
        Q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        K = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        V = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)

        if self.use_alibi:
            bias = self.alibi(T, T)  # (1, n_head, T, T)
            bias = bias.expand_as(scores)  # 从(1, n_head, T, T)复制为(B, n_head, T, T)
            #print(bias)
            scores = scores + bias  # 广播加法

        attn = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.out(out)  # (B, T, n_embd)
        return out, attn

# masked multi head attention
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_alibi=False, max_seq_len=512):
        super(MaskedMultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_alibi = use_alibi

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        if self.use_alibi:
            self.alibi = AliBi_decoder(max_seq_len, n_head, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        B, T, C = x.size()
        Q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        K = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        V = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)

        if self.use_alibi:
            bias = self.alibi(T, T)  # (1, n_head, T, T)
            bias = bias.expand_as(scores)  # 从(1, n_head, T, T)复制为(B, n_head, T, T)
            scores = scores + bias  # 广播加法

        # 创建掩蔽矩阵
        mask = torch.tril(torch.ones((T, T), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.out(out)  # (B, T, n_embd)
        return out, attn

# feed forward network
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 100),  # 隐藏层维度为100
            nn.ReLU(),
            nn.Linear(100, n_embd)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))

# transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1, use_alibi=False, max_seq_len=512, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.use_alibi = use_alibi

        if self.use_alibi:
            self.alibi = AliBi(max_seq_len, n_head, device=device)

        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, use_alibi=self.use_alibi) for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)  # (B, T, n_embd)
        position = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(position)  # (1, T, n_embd)
        x = self.dropout(token_emb + pos_emb)  # (B, T, n_embd)

        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)
        
        x = self.layer_norm(x)
        x_mean = x.mean(dim=1)  # (B, n_embd)
        x_mean = x_mean.float()  # 确保类型为 Float
        return x_mean, attention_maps

# transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1, use_alibi=False, max_seq_len=512):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 移除位置编码，因为使用 AliBi
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(n_embd, n_head, dropout, use_alibi=True, max_seq_len=max_seq_len) 
            for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.size()
        x = self.token_embedding(x)  # (B, T, n_embd)
        x = self.dropout(x)  # 移除位置编码的加法

        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attention_maps.append(attn)
        
        x = self.layer_norm(x)
        logits = self.output_layer(x)
        return logits, attention_maps
# tra
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_alibi=False):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, dropout, use_alibi=use_alibi)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn = self.attention(self.layer_norm1(x))
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.layer_norm2(x))
        x = x + self.dropout(ff_out)
        return x, attn

# 定义 Transformer 解码器块（带掩蔽自注意力）
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_alibi=False, max_seq_len=512):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MaskedMultiHeadAttention(n_embd, n_head, dropout, use_alibi, max_seq_len)
        self.encoder_attention = MultiHeadAttention(n_embd, n_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.layer_norm3 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs=None):
        # masked self attention
        attn_out, attn = self.masked_attention(self.layer_norm1(x))
        x = x + self.dropout(attn_out)

        if encoder_outputs is not None:
            # encoder-decoder attention
            enc_attn_out, enc_attn = self.encoder_attention(self.layer_norm2(x), encoder_outputs)
            x = x + self.dropout(enc_attn_out)

        # feed forward network
        ff_out = self.feed_forward(self.layer_norm3(x))
        x = x + self.dropout(ff_out)
        return x, attn
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_alibi=True, max_seq_len=512):
        super(TransformerDecoderBlock, self).__init__()
        # 在掩蔽自注意力中使用 AliBi
        self.masked_attention = MaskedMultiHeadAttention(
            n_embd, n_head, dropout, use_alibi=True, max_seq_len=max_seq_len
        )
        # 在交叉注意力中不使用 AliBi
        self.encoder_attention = MultiHeadAttention(
            n_embd, n_head, dropout, use_alibi=False
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.layer_norm3 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs=None):
        attn_out, attn = self.masked_attention(self.layer_norm1(x))
        x = x + self.dropout(attn_out)

        if encoder_outputs is not None:
            enc_attn_out, enc_attn = self.encoder_attention(self.layer_norm2(x), encoder_outputs)
            x = x + self.dropout(enc_attn_out)

        ff_out = self.feed_forward(self.layer_norm3(x))
        x = x + self.dropout(ff_out)
        return x, attn