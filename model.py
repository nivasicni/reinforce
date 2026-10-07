import torch
from torch import nn
from torch.nn import functional as F


class TransformerBlock(nn.Module):
    """attention + residual, then MLP + residual.
    B = batch size, L = sequence length, D = model width,
    H = number of heads, and d = D / H (default: D=128, H=4, d=32).
    both sublayers preserve [B, L, D]
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_norm = nn.LayerNorm(d_model)

        # compute Q, K, V together
        self.qkv = nn.Linear(d_model, 3 * d_model)

        self.attention_output = nn.Linear(d_model, d_model)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            # shared across positions
            # D -> 4D -> D
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D].
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x)) # broadcast to each batch, each position

        # [B, L, 3D] -> [B, L, 3, H, d]
        qkv = qkv.reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        #  [3, B, H, L, d] -> q, k, v:  [B, H, L, d].

        # per head: softmax(Q K^T / sqrt(d) + causal mask) V.
        # scores: [B, H, L, L]; output: [B, H, L, d].
        # position i can attend to positions 0..i, never future positions.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)


        # [B, H, L, d] -> [B, L, H, d] -> [B, L, D].
        attended = attended.transpose(1, 2).reshape(batch, length, width)

        x = x + self.attention_output(attended)
        return x + self.mlp(self.mlp_norm(x))


class DigitTransformer(nn.Module):

    def __init__(
        self, max_seq_len: int, d_model: int = 128,
        n_heads: int = 4, n_layers: int = 2,
    ):
        super().__init__()
        if min(max_seq_len, d_model, n_heads, n_layers) < 1:
            raise ValueError("Model dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(10, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        # final hidden vector -> 10 logits for next digit
        self.lm_head = nn.Linear(d_model, 10, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits [batch, sequence, 10]."""
        length = tokens.size(1)
        if not 1 <= length <= self.max_seq_len:
            raise ValueError("Input length exceeds the model context or is empty")
        # tokens: [B, L], positions: [L], Position embeddings [L, D]
        # broadcast across the batch when added to token embeddings [B, L, D]
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.lm_head(self.final_norm(self.blocks(x)))

    @torch.no_grad()
    def generate(
        self, prompts: torch.Tensor, output_length: int, *, do_sample: bool = False,
    ) -> torch.Tensor:
        """Return fixed-length new tokens, greedily or sampled from the policy
        """
        if output_length < 1:
            raise ValueError("output_length must be positive")
        # prompts: [batch_size, input_length]
        # for example [128, 6] for modular addition.
        # each row holds digit IDs; [1, 0, 5, 0, 6, 9]  - "105069"
        # prompts.size(1) is the prompt length (6).
        # final chosen token doesnt get fed back
        if prompts.size(1) + output_length - 1 > self.max_seq_len:
            raise ValueError("Generation exceeds the model context")
        tokens = prompts
        for _ in range(output_length):
            logits = self(tokens)                   # [B, L, 10]
            next_digit_logits = logits[:, -1]       # [B, 10]
            if do_sample:
                chosen = torch.distributions.Categorical(
                    logits=next_digit_logits
                ).sample().unsqueeze(-1)            # [B, 1]
            else:
                chosen = next_digit_logits.argmax(
                    dim=-1, keepdim=True
                )                                   # [B, 1]
            tokens = torch.cat((tokens, chosen), dim=1)
        # we return only the answer
        return tokens[:, prompts.size(1):]
