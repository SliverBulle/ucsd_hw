import torch
import torch.nn as nn
import math


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

# define Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
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

# define Transformer decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(n_embd, vocab_size)  # 输出层，预测下一个词

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
        logits = self.output_layer(x)  # (B, T, vocab_size)
        return logits, attention_maps

# define Transformer encoder block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, dropout)
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

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MaskedMultiHeadAttention(n_embd, n_head, dropout)
        self.encoder_attention = MultiHeadAttention(n_embd, n_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.layer_norm3 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs=None):
        # 掩蔽自注意力
        attn_out, attn = self.masked_attention(self.layer_norm1(x))
        x = x + self.dropout(attn_out)

        if encoder_outputs is not None:
            # 编码器-解码器注意力
            enc_attn_out, enc_attn = self.encoder_attention(self.layer_norm2(x), encoder_outputs)
            x = x + self.dropout(enc_attn_out)

        # 前馈网络
        ff_out = self.feed_forward(self.layer_norm3(x))
        x = x + self.dropout(ff_out)
        return x, attn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super(MaskedMultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        Q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        K = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        V = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_head, T, T)

        # 创建掩蔽矩阵
        mask = torch.tril(torch.ones((T, T), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)
        attn = self.dropout(attn)
        out = attn @ V  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.out(out)  # (B, T, n_embd)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out=None):
        B, T, C = x.size()
        Q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        K = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        V = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        attn = torch.softmax(scores, dim=-1)  # (B, n_head, T, T)
        attn = self.dropout(attn)
        out = attn @ V  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.out(out)  # (B, T, n_embd)
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 100),  # hidden layer dimension is 100
            nn.ReLU(),
            nn.Linear(100, n_embd)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))