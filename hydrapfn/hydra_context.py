from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

import math
import copy
from functools import partial
from hydra.modules.hydra import Hydra
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.mha import MHA

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



#
# Copied the Block from the official Mamba repository here and edited it for Hydra. Hydra does not have inference params so they are not included.
#
class HydraBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    '''
    Function from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py - Here used to implement hydra.
    '''

    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:

        #print(f"SSM cfg: {ssm_cfg}")
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        mixer_cls = partial(
            Hydra,
            layer_idx=layer_idx,
            #**ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )

    block = HydraBlock(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )

    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    '''
    Function from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    '''
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class HydraBackbone(nn.Module):
    '''
    Backbone blocks of the Hydra Model.
    '''
    def __init__(self,
                d_model: int,
                num_layers: int,
                ssm_config=None,
                norm_epsilon: float = 1e-5,
                rms_norm: bool = False,
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                attn_layer_idx=[],
                attn_config={},
                d_intermediate=0
                ):
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.blocks = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_config,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_config,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )


    def forward(
        self,
        x,
        inference_parameters=None
    ):
            
        hidden_states = x

        residual = None

        #for block in self.blocks: hidden_states, residual = block(hidden_states, residual, inference_params=inference_parameters)
        for block in self.blocks: hidden_states, residual = block(hidden_states, residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        return hidden_states


# =========================
# CROSS ATTENTION (FIXED)
# =========================

class SimpleCrossAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, query, context):
        query = self.norm_q(query)
        context = self.norm_kv(context)

        B, Nq, D = query.shape
        Nc = context.shape[1]

        Q = self.q_proj(query).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, Nc, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, Nc, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)

        return query + self.out_proj(out)


# =========================
# MODEL
# =========================

class HydraModel(nn.Module):
    def __init__(
        self,
        encoder,
        y_encoder,
        n_out,
        ninp,
        nhid,
        num_layers=1,
        cross_attention_mode="single",  # "single", "dual_sum", "dual_concat"
        num_heads=4,
        device="cpu",
        dtype=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.y_encoder = y_encoder
        self.mode = cross_attention_mode

        # ---- backbone (same style as before) ----
        self.backbone = HydraBackbone(
            d_model=ninp,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

        # ---- cross attention ----
        self.cross_attn = SimpleCrossAttention(ninp, num_heads)

        if self.mode in ["dual_sum", "dual_concat"]:
            self.cross_attn_raw = SimpleCrossAttention(ninp, num_heads)

        if self.mode == "dual_concat":
            self.fusion = nn.Sequential(
                nn.Linear(2 * ninp, ninp),
                nn.GELU(),
            )

        # ---- alignment (IMPORTANT FIX) ----
        self.query_proj = nn.Linear(ninp, ninp)
        self.context_proj = nn.Linear(ninp, ninp)

        # ---- decoder ----
        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_out),
        )

    # =========================
    # HELPERS
    # =========================

    def _encode(self, x, y):
        x = self.encoder(x)
        y = self.y_encoder(y.unsqueeze(-1) if y.dim() < x.dim() else y)
        return x, y

    def _permute_context(self, x, y, split_pos):
        device = x.device
        perm = torch.randperm(split_pos, device=device)

        x_perm = x.clone()
        y_perm = y.clone()

        x_perm[:split_pos] = x_perm[perm]
        y_perm[:split_pos] = y_perm[perm]

        return x_perm, y_perm

    def _split(self, x, y, split_pos):
        context = x[:split_pos] + y[:split_pos]
        query = x[split_pos:]

        return context.permute(1, 0, 2), query.permute(1, 0, 2)

    def _align(self, query, context):
        return self.query_proj(query), self.context_proj(context)

    def _cross_attend(self, query, context, raw_context):
        attn_hidden = self.cross_attn(query, context)

        if self.mode == "single":
            return attn_hidden

        attn_raw = self.cross_attn_raw(query, raw_context)

        if self.mode == "dual_sum":
            return (attn_hidden + attn_raw) / math.sqrt(2)

        if self.mode == "dual_concat":
            return self.fusion(torch.cat([attn_hidden, attn_raw], dim=-1))

        raise ValueError(self.mode)

    def _decode(self, x):
        return self.decoder(x).permute(1, 0, 2)

    # =========================
    # TRAIN
    # =========================

    def forward_train(self, src, split_pos, compute_perm_reg=False):
        _, x, y = src

        # ---- encode ----
        x, y = self._encode(x, y)

        # ---- main path ----
        context, query = self._split(x, y, split_pos)

        context_hidden = self.backbone(context)

        query, context_hidden = self._align(query, context_hidden)

        conditioned = self._cross_attend(
            query,
            context_hidden,
            raw_context=context,
        )

        output = self._decode(conditioned)

        # ---- permutation regularization ----
        perm_context_hidden = None
        if compute_perm_reg:
            x_perm, y_perm = self._permute_context(x, y, split_pos)

            context_perm, _ = self._split(x_perm, y_perm, split_pos)

            perm_context_hidden = self.backbone(context_perm)

        return output, context_hidden, perm_context_hidden

    # =========================
    # INFERENCE 
    # =========================

    def forward_inference(self, src, split_pos, num_pcps=1):
        if num_pcps == 1:
            return self.forward_train(src, split_pos, compute_perm_reg=False)

        _, x, y = src

        x, y = self._encode(x, y)
        context, query = self._split(x, y, split_pos)

        B, S, D = context.shape
        Sq = query.shape[1]
        P = num_pcps

        # ---- permutations ----
        perms = torch.stack(
            [torch.randperm(S, device=context.device) for _ in range(P)]
        )

        context_exp = context.unsqueeze(0).expand(P, B, S, D)
        perms_exp = perms.unsqueeze(1).unsqueeze(-1).expand(P, B, S, D)

        context_perm = torch.gather(context_exp, 2, perms_exp)
        context_perm = context_perm.reshape(P * B, S, D)

        # ---- backbone ----
        context_hidden = self.backbone(context_perm)

        # ---- expand query ----
        query_exp = query.unsqueeze(0).expand(P, B, Sq, D).reshape(P * B, Sq, D)
        raw_context_exp = context.unsqueeze(0).expand(P, B, S, D).reshape(P * B, S, D)

        # ---- align ----
        query_exp, context_hidden = self._align(query_exp, context_hidden)

        # ---- cross attention ----
        conditioned = self._cross_attend(
            query_exp,
            context_hidden,
            raw_context=raw_context_exp,
        )

        # ---- decode ----
        decoded = self.decoder(conditioned)  # (P*B, Sq, ...)
        decoded = decoded.reshape(P, B, Sq, -1)

        output = decoded.mean(dim=0).permute(1, 0, 2)

        context_hidden = context_hidden.reshape(P, B, S, D).mean(dim=0)

        return output, context_hidden, None

    # =========================
    # MAIN
    # =========================

    def forward(
        self,
        src,
        single_eval_pos,
        inference=False,
        num_pcps=1,              # <-- restored
        compute_perm_reg=False,
    ):
        if inference:
            return self.forward_inference(
                src,
                single_eval_pos,
                num_pcps=num_pcps
            )

        return self.forward_train(
            src,
            single_eval_pos,
            compute_perm_reg=compute_perm_reg
        )