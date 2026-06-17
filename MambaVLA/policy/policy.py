import torch
import torch.nn as nn
import einops
import math
from typing import Optional, Any
import logging
logger = logging.getLogger(__name__)



class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)

        if len(t_emb.shape) == 2:
            t_emb = einops.rearrange(t_emb, 'b d -> b 1 d')

        return t_emb


class SigmaFiLMLayer(nn.Module):
    """Apply FiLM-style modulation to a sequence using the sigma embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale = nn.Linear(embed_dim, embed_dim)
        self.shift = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, inputs: torch.Tensor, sigma_embedding: torch.Tensor) -> torch.Tensor:
        if sigma_embedding.dim() == 3:
            cond = sigma_embedding[:, 0, :]
        elif sigma_embedding.dim() == 2:
            cond = sigma_embedding
        else:
            cond = sigma_embedding.view(sigma_embedding.size(0), -1)

        scale = self.scale(cond).unsqueeze(1)
        shift = self.shift(cond).unsqueeze(1)
        return inputs * (1 + scale) + shift


class ActionQueryDecoder(nn.Module):
    """Lightweight decoder that lets action queries re-attend to encoder memories."""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_action_tokens: bool = True,
    ):
        super().__init__()

        self.use_action_tokens = use_action_tokens
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

    def forward(
        self,
        memory: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        sigma_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = memory.size(0)
        query = self.query_embed.expand(batch_size, -1, -1)

        if self.use_action_tokens and action_tokens is not None:
            query = query + action_tokens

        if sigma_embedding is not None:
            if sigma_embedding.dim() == 3:
                sigma_tokens = sigma_embedding.expand(-1, query.size(1), -1)
            else:
                sigma_tokens = sigma_embedding.unsqueeze(1).expand(-1, query.size(1), -1)
            query = query + sigma_tokens

        attn_output = self.self_attn(query, query, query, need_weights=False)[0]
        query = self.norm1(query + self.dropout1(attn_output))

        cross_output = self.cross_attn(query, memory, memory, need_weights=False)[0]
        query = self.norm2(query + self.dropout2(cross_output))

        ffn_output = self.ffn(query)
        query = self.norm3(query + self.dropout3(ffn_output))

        return query


class MambaVLAPolicy(nn.Module):
    def __init__(
            self,
            encoder: Any,
            latent_dim: int,
            action_dim: int,
            lang_emb_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            lang_tok_len: int,
            obs_tok_len: int,
            action_seq_len: int,
            linear_output: bool = False,
            use_ada_conditioning: bool = False,
            use_pos_emb: bool = True,
            use_sigma_film: bool = False,
            use_action_decoder: bool = False,
            action_decoder_heads: int = 4,
            action_decoder_mlp_ratio: float = 2.0,
            action_decoder_dropout: float = 0.0,
            decoder_use_action_tokens_as_queries: bool = True
    ):
        super().__init__()

        self.encoder = encoder

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            lang_tok_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = lang_tok_len + obs_tok_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(latent_dim, embed_dim)

        # linear embedding for the goal
        self.lang_emb = nn.Linear(lang_emb_dim, embed_dim)

        # linear embedding for the action
        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.sigma_emb = TimeEmbedding(embed_dim)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            # position embedding
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.action_dim = action_dim
        self.obs_dim = latent_dim
        self.embed_dim = embed_dim

        self.lang_tok_len = lang_tok_len
        self.obs_tok_len = obs_tok_len
        self.action_seq_len = action_seq_len

        self.use_ada_conditioning = use_ada_conditioning
        self.use_sigma_film = use_sigma_film
        self.use_action_decoder = use_action_decoder

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        if self.use_sigma_film:
            self.state_film = SigmaFiLMLayer(embed_dim)
            self.action_film = SigmaFiLMLayer(embed_dim)
            self.output_film = SigmaFiLMLayer(embed_dim)
            if self.goal_conditioned:
                self.goal_film = SigmaFiLMLayer(embed_dim)

        if self.use_action_decoder:
            self.action_decoder = ActionQueryDecoder(
                embed_dim=embed_dim,
                num_queries=self.action_seq_len,
                num_heads=action_decoder_heads,
                mlp_ratio=action_decoder_mlp_ratio,
                dropout=action_decoder_dropout,
                use_action_tokens=decoder_use_action_tokens_as_queries
            )
        else:
            self.action_decoder = None

        self.apply(self._init_weights)

        if self.use_sigma_film:
            self._reset_film_parameters()

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _reset_film_parameters(self):
        film_layers = [self.state_film, self.action_film, self.output_film]
        if self.goal_conditioned:
            film_layers.append(self.goal_film)

        for layer in film_layers:
            layer.reset_parameters()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states,
            actions,
            lang_cond,
            sigma
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=states.device, dtype=states.dtype)
        else:
            sigma = sigma.to(states.device)

        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        elif sigma.dim() > 1:
            sigma = sigma.view(sigma.size(0))

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            lang_embed = self.lang_emb(lang_cond)
            if self.use_sigma_film:
                lang_embed = self.goal_film(lang_embed, emb_t)
            if self.use_pos_emb:
                if lang_embed is not None:
                    if lang_embed.dim() == 2:
                        lang_embed = lang_embed.unsqueeze(1)
                    elif lang_embed.dim() == 1:
                        lang_embed = lang_embed.unsqueeze(0).unsqueeze(1)

                lang_embed += self.pos_emb[:, :self.lang_tok_len, :]
            goal_x = self.drop(lang_embed)

        state_embed = self.tok_emb(states)
        if self.use_sigma_film:
            state_embed = self.state_film(state_embed, emb_t)
        if self.use_pos_emb:
            state_embed += self.pos_emb[:, self.lang_tok_len:(self.lang_tok_len + t), :]
        state_x = self.drop(state_embed)

        action_embed = self.action_emb(actions)
        if self.use_sigma_film:
            action_embed = self.action_film(action_embed, emb_t)
        if self.use_pos_emb:
            action_embed += self.pos_emb[:, (self.lang_tok_len + t):(self.lang_tok_len + t + t_a), :]
        action_x = self.drop(action_embed)

        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        if self.use_action_decoder:
            context_len = self.seq_size - self.action_seq_len
            context_with_time = context_len + 1
            encoder_memory = encoder_output[:, :context_with_time, :]
            action_tokens = encoder_output[:, -self.action_seq_len:, :]
            decoded_actions = self.action_decoder(
                encoder_memory,
                action_tokens=action_tokens,
                sigma_embedding=emb_t
            )
        else:
            decoded_actions = encoder_output[:, -self.action_seq_len:, :]

        if self.use_sigma_film:
            decoded_actions = self.output_film(decoded_actions, emb_t)

        pred_actions = self.action_pred(decoded_actions)

        return pred_actions