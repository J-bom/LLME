import os
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


#end of text token id (native <|endoftext|> for the cl100k_base)
EOT_ID = 100257

#start of input token (not native <|im_start|> token added using SFT)
IM_START_ID = 100277

#end of input token (not native <|im_end|> token added using SFT)
IM_END_ID = 100278

#total number of unique tokens in the tokenizer padded - must match loaded model checkpoint
VOCAB_SIZE = 100352

#number of Transformer blocks (depth) - must match loaded model checkpoint stats
LAYERS = 24

#Hidden layer embedding dimension (width) - must match loaded model checkpoint stats
DIMENSIONS = 1024

#number of Query attention heads - must match loaded model checkpoint stats
N_HEAD = 16

#number of Key/Value heads (for GQA) - must match loaded model checkpoint stats
KV_HEAD_NUM = 4

#Maximum sequence length (context window)
BLOCK_SIZE = 2048

#model default system prompt
DEFAULT_SYSTEM = (
    "You are Alter Ego, a small AI built from scratch. You're casual and direct. "
    "You're not great with facts, math, or current events — when you don't know "
    "something, just say so. You're better at chatting than at answering questions."
)



def get_tokenizer():
    """
    extends cl100k_base to include <|im_start|> and <|im_end|>.
    :return: built tokenizer
    """
    base = tiktoken.get_encoding("cl100k_base")
    return tiktoken.Encoding(
        name="cl100k_alterego",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={
            **base._special_tokens,
            "<|im_start|>": IM_START_ID,
            "<|im_end|>": IM_END_ID,
        },
    )



def precompute_freqs(dim, end, theta=10000.0):
    """
    Precomputes frequency constants for Rotary Positional Embeddings (RoPE).
    :param dim: internal embedding dimension
    :param end: max context len
    :param theta: scaling factor for the frequency base
    :return: a tuple of (cos, sin) tensors for position encoding
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """
    Applies Rotary Positional Embeddings (RoPE) to query and key tensors.
    Rotates token embeddings using precomputed frequency buffers to inject positional information.
    :return: (xq_out, xk_out) tensors with integrated position data
    """
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _repeat_kv(x, n_rep):
    """
    repeats Key and Value heads to match the number of Query heads.
    :return: expanded x tensor with shape (bs, slen, n_q_heads, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Provides input scaling stabilization without re-centering (no bias), improving computational efficiency.
    :return: normalized and re-scaled tensor
    """
    def __init__(self, dim, eps=1e-6):
        """
        initiallizer for the RMSNorm layer.
        :param dim: embedding dimension to be normalized
        :param eps: epsilon value for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Performs the forward pass: normalizes input by its RMS and scales by weight.
        :param x: input tensor (batch_size, seq_len, dim)
        :return: normalized tensor cast back to the input's original dtype
        """
        dtype_in = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype_in)


class CasualSelfAttention(nn.Module):
    """
    Implements Grouped-Query Attention (GQA) with Rotary Positional Embeddings (RoPE).
    Handles causal self-attention in a casual way for autoregressive generation, including KV cache support.
    """
    def __init__(self, config):
        """
        self attention initallizer. initializes attention layers and projection matrices.
        Sets up head dimensions and GQA groups based on the model configuration.
        :param config: model config
        """
        super().__init__()
        self.head_num = config.head_num
        self.kv_head_num = config.kv_head_num
        self.dimention_num = config.dimention_num
        self.head_dim = self.dimention_num // self.head_num
        assert self.head_num % self.kv_head_num == 0
        self.num_key_value_groups = self.head_num // self.kv_head_num
        self.q_size = self.head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.c_attn = nn.Linear(self.dimention_num, self.q_size + 2 * self.kv_size, bias=False)
        self.c_proj = nn.Linear(self.dimention_num, self.dimention_num, bias=False)
        self.c_proj.LLME_SCALE_INIT = 1

    def forward(self, x, freqs_cos, freqs_sin, use_cache=False, past_kv=None):
        """
        Executes the attention mechanism: QKV projection, RoPE application, and GQA.
        :param x: input tensor of shape (batch, seq_len, dim)
        :param freqs_cos: precomputed cosine frequencies for RoPE
        :param freqs_sin: precomputed sine frequencies for RoPE
        :param use_cache: if True, returns current KV states for future use
        :param past_kv: optional tuple of (k, v) from previous steps to speed up inference
        :return: tuple of (output, present_kv)
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=2)
        q = q.view(B, T, self.head_num, self.head_dim)
        k = k.view(B, T, self.kv_head_num, self.head_dim)
        v = v.view(B, T, self.kv_head_num, self.head_dim)

        if past_kv is None:
            seq_cos = freqs_cos[:T]
            seq_sin = freqs_sin[:T]
        else:
            offset = past_kv[0].size(2)
            seq_cos = freqs_cos[offset: offset + T]
            seq_sin = freqs_sin[offset: offset + T]
        q, k = apply_rotary_emb(q, k, seq_cos, seq_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        k = _repeat_kv(k.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
        v = _repeat_kv(v.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)

        causal = T > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, present_kv


class MLP(nn.Module):
    """
    the multi layer perceptron. implements a SwiGLU feed-forward network.
    """
    def __init__(self, config):
        """
        initallizer for the ml class, initializes the MLP layers with hidden dimension scaling.
        Aligns hidden_dim to the nearest multiple of 128 for hardware optimization.
        :param config: model config
        """
        super().__init__()
        hidden_dim = int(8 * config.dimention_num / 3)
        hidden_dim = ((hidden_dim + 127) // 128) * 128
        self.w1 = nn.Linear(config.dimention_num, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.dimention_num, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.dimention_num, bias=False)
        self.w3.LLME_SCALE_INIT = 1

    def forward(self, x):
        """
        executes the SwiGLU gating mechanism: w3(SiLU(w1(x)) * w2(x)).
        :param x: input tensor from the attention block
        :return: processed features of the same shape as input
        """
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.w3(gate * up)


class Block(nn.Module):
    """
    A simple transformer layer aka block
    combines the RMSNorm, self atention and mlp with residual connections
    """
    def __init__(self, config):
        """
        initiallizer for the block class, initallizes the block components
        :param config: model config
        """
        super().__init__()
        self.layernorm1 = RMSNorm(config.dimention_num)
        self.attn = CasualSelfAttention(config)
        self.layernorm2 = RMSNorm(config.dimention_num)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cos, freqs_sin, use_cache=False, past_kv=None):
        """
        Performs a full Transformer layer pass:
        Norm -> Attention -> Add -> Norm -> MLP -> Add.

        :param x: input tensor (batch, seq_len, dim)
        :param freqs_cos/sin: RoPE frequency buffers
        :param use_cache: whether to return KV cache
        :param past_kv: previous KV state for inference
        :return: (output_tensor, present_kv)
        """
        attn_out, present_kv = self.attn(self.layernorm1(x), freqs_cos, freqs_sin, use_cache, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.layernorm2(x))
        return x, present_kv


@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = VOCAB_SIZE
    layer_num: int = LAYERS
    head_num: int = N_HEAD
    kv_head_num: int = KV_HEAD_NUM
    dimention_num: int = DIMENSIONS


class GPT(nn.Module):
    """
        The full GPT Language Model.
        integrates embedding, a stack of transformer blocks, and a linear head
        for next-token prediction. features weight Tying and RoPE frequency caching.
        """
    def __init__(self, config):
        """
        Performs a full Transformer layer pass:
        Norm -> Attention -> Add -> Norm -> MLP -> Add.

        :param x: input tensor (batch, seq_len, dim)
        :param freqs_cos/sin: RoPE frequency buffers
        :param use_cache: whether to return KV cache
        :param past_kv: previous KV state for inference
        :return: (output_tensor, present_kv)
        """
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.dimention_num),
            h=nn.ModuleList([Block(config) for _ in range(config.layer_num)]),
            layernorm_f=RMSNorm(config.dimention_num),
        ))
        self.lmhead = nn.Linear(config.dimention_num, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lmhead.weight

        head_dim = config.dimention_num // config.head_num
        cos, sin = precompute_freqs(head_dim, config.block_size)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, idx, use_cache=False, past_kvs=None):
        """
        Performs a full model pass. Iterates through all blocks, managing KV cache states for
        efficient autoregressive generation.

        :param idx: input tensor of token indices (batch, seq_len)
        :param use_cache: whether to compute and return KV cache for inference
        :param past_kvs: list of (k, v) tuples from previous steps
        :return: (logits, present_kvs) where logits are the raw predictions
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq len {T} > block size {self.config.block_size}"

        x = self.transformer.wte(idx)
        present_kvs = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(x, self.freqs_cos, self.freqs_sin, use_cache, past_kv)
            if use_cache:
                present_kvs.append(present_kv)

        x = self.transformer.layernorm_f(x)
        logits = self.lmhead(x)
        return logits, present_kvs


class AlterEgo:
    """High level inference wrapper for AlterEgo"""

    def __init__(self, checkpoint_path, device=None):
        """
        initiallzer for the inference class. initiallizes and loads the model and related componentes
        :param checkpoint_path: model path
        :param device: device to load the model and operate on
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = GPT(GPTConfig()).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.tokenizer = get_tokenizer()

    def prompt(self, user_message, history=None, system_prompt=DEFAULT_SYSTEM, max_new_tokens=200, temperature=0.7, top_k=50, top_p=1.0, repetition_penalty=1.1,):
        """
        The main entry point for interaction.
        Formats the conversation history and triggers the generation process.
        :param user_message: the latest input string from the user
        :param history: list of prior (role, content) tuples
        :param system_prompt: persona/instructions for the model
        :return: processed assistant response string
        """
        turns = list(history) if history else []
        turns.append(("user", user_message))

        prompt_tokens = self.render(system_prompt, turns)

        return self.generate(prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,).strip()

    def render(self, system_prompt, turns):
        """
        Converts a conversation into a single prompt string using the ChatML format.
        Wraps messages in <|im_start|> and <|im_end|> special tokens.
        :return: list of encoded token IDs
        """
        parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
        for role, content in turns:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return self.tokenizer.encode(
            "".join(parts),
            allowed_special={"<|im_start|>", "<|im_end|>"},
            disallowed_special=(),
        )

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty,):
        """
        The inference loop. Predicts tokens one-by-one using the KV cache.
        Implements sampling strategies: Temperature, Top-K, Top-P, and Repetition Penalty.
        :return: decoded string of generated tokens
        """
        max_prompt_len = BLOCK_SIZE - max_new_tokens - 1
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[-max_prompt_len:]

        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        prompt_len = idx.size(1)
        generated_ids = []

        logits, past_kvs = self.model(idx, use_cache=True, past_kvs=None)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()

            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                recent = torch.tensor(generated_ids[-64:], device=self.device)
                unique = torch.unique(recent)
                sel = next_logits[0, unique]
                sel = torch.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
                next_logits[0, unique] = sel

            if temperature != 1.0 and temperature != 0.0:
                next_logits = next_logits / max(temperature, 1e-5)

            if top_k and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')

            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumprobs > top_p
                mask[:, 0] = False
                indices_to_remove = sorted_idx[mask]
                next_logits[0, indices_to_remove] = -float('Inf')

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tok_id = next_token.item()

            if tok_id == IM_END_ID or tok_id == EOT_ID:
                break
            if prompt_len + len(generated_ids) + 1 >= BLOCK_SIZE:
                break

            generated_ids.append(tok_id)

            logits, past_kvs = self.model(next_token, use_cache=True, past_kvs=past_kvs)

        return self.tokenizer.decode(generated_ids)