from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

from torch import nn
import math
import torch
import copy
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from hydra.modules.hydra import Hydra
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.mha import MHA

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from hydrapfn.tabicl_embedding.embedding import ColEmbedding
from hydrapfn.tabicl_embedding.interaction import RowInteraction


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


    def forward(self,
                x,
                inference_parameters
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


class HydraFullModel(nn.Module):
    '''
    Full-sequence Hydra Model with optional ColEmbedding and RowInteraction.
    Applies Hydra over the entire sequence (train + test), like hydra_old,
    but with optional tabular embeddings from hydra_icl.
    '''

    def __init__(self,
                 encoder,
                 n_out,
                 ninp,
                 nhid,
                 num_layers: int = 1,
                 ssm_config = None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 y_encoder=None,
                 initializer_config = None,
                 fused_add_norm = False,
                 residual_in_fp32 = False,
                 device = "cpu",
                 dtype=None,
                 use_col_embedding=False,
                 col_max_classes: int = 10,
                 col_num_blocks: int = 3,
                 col_nhead: int = 4,
                 col_num_inds: int = 128,
                 col_affine: bool = False,
                 col_feature_group="same",
                 col_feature_group_size: int = 3,
                 col_target_aware: bool = True,
                 col_ssmax="qassmax-mlp-elementwise",
                 col_num_cls: int = 4,
                 col_ff_factor: int = 2,
                 col_dropout: float = 0.0,
                 ) -> None:

        super().__init__()
        self.model_type = 'hydra-full'
        self.use_col_embedding = use_col_embedding
        self.device = device
        self.dtype = dtype

        if use_col_embedding:
            # ninp must equal col_embed_dim * col_num_cls.
            # e.g. ninp=512, col_num_cls=4 -> col_embed_dim=128
            col_embed_dim = ninp // col_num_cls

            # ColEmbedding: (B, T, H) -> (B, T, H, col_embed_dim)  [per-cell embeddings]
            self.col_embedder = ColEmbedding(
                embed_dim=col_embed_dim,
                num_blocks=col_num_blocks,
                nhead=col_nhead,
                num_inds=col_num_inds,
                dim_feedforward=col_embed_dim * col_ff_factor,
                dropout=col_dropout,
                activation="gelu",
                norm_first=True,
                bias_free_ln=False,
                affine=col_affine,
                feature_group=col_feature_group,
                feature_group_size=col_feature_group_size,
                target_aware=col_target_aware,
                max_classes=col_max_classes,
                reserve_cls_tokens=col_num_cls,
                ssmax=col_ssmax,
                recompute=False,
            )

            # RowInteraction: (B, T, H, col_embed_dim) -> (B, T, ninp)
            self.row_interactor = RowInteraction(
                embed_dim=col_embed_dim,
                num_blocks=3,           # matches TabICL default
                nhead=col_nhead,
                dim_feedforward=col_embed_dim * col_ff_factor,
                num_cls=col_num_cls,
                rope_base=100000,       # TabICL default
                rope_interleaved=False, # TabICL default
                dropout=col_dropout,
                activation="gelu",
                norm_first=True,
                bias_free_ln=False,
                recompute=False,
            )

            self.col_num_cls = col_num_cls
            self.encoder = None
            self.y_encoder = None
        else:
            self.encoder = encoder
            self.y_encoder = y_encoder
            self.col_embedder = None
            self.row_interactor = None

        self.num_layers = num_layers
        self.ssm_config = ssm_config
        self.ssm_config = {"layer": "Hydra"}       # Specify the mamba version used
        self.rms_norm = rms_norm
        self.initializer_config = initializer_config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        factory_kwargs = {"device": device, "dtype": dtype}

        self.mamba_backbone = HydraBackbone(
            d_model=ninp,
            num_layers=self.num_layers,
            ssm_config=self.ssm_config,
            norm_epsilon=1e-5,
            rms_norm=False,     # Doesn't work with true yet.
            initializer_cfg=self.initializer_config,
            fused_add_norm=self.fused_add_norm,
            residual_in_fp32=self.residual_in_fp32,
            device=self.device,
            dtype=self.dtype
        )

        self.decoder = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out))

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            ninp, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=num_layers,
                **(initializer_config if initializer_config is not None else {}),
            )
        )

    def _encode(self, x, y, single_eval_pos):
        """Encode inputs into per-row representations.

        When use_col_embedding=True:
            Pipeline: ColEmbedding -> RowInteraction
            - ColEmbedding: (B, T, H) -> (B, T, H, col_embed_dim)  per-cell embeddings
            - RowInteraction: (B, T, H, col_embed_dim) -> (B, T, ninp)  aggregates
            - Output is permuted to (T, B, ninp)

        When use_col_embedding=False:
            Falls back to the original linear encoder + y_encoder logic.
        """
        if self.use_col_embedding:
            x_bf = x.permute(1, 0, 2)          # (B, T, H)
            y_bf = y.permute(1, 0)             # (B, T)

            # Step 1: column-wise embedding — per-cell representations
            # Output: (B, T, H, col_embed_dim)
            col_encoded = self.col_embedder(
                x_bf,
                y_train=y_bf[:, :single_eval_pos],  # only context labels for target-aware
                embed_with_test=True,  # include test rows
            )

            # Step 2: row-wise interaction — aggregate H features into ninp per row
            # Output: (B, T, ninp)
            encoded = self.row_interactor(col_encoded)

            encoded = encoded.permute(1, 0, 2)  # (T, B, ninp)
            return encoded, None
        else:
            x_enc = self.encoder(x)
            y_enc = self.y_encoder(y.unsqueeze(-1) if len(y.shape) < len(x.shape) else y)
            return x_enc, y_enc

    def forward(self, src: tuple, single_eval_pos: int, **kwargs):

        _, x_src, y_src = src               # Split input into style, train (x) and test (y) part.

        # Encode inputs
        x_src, y_src = self._encode(x_src, y_src, single_eval_pos)

        if self.use_col_embedding:
            # When using col embedding, y is already incorporated
            src = x_src
        else:
            # Original logic: combine x and y for training part
            train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
            src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        # Before: BPTT, (batch_size / aggregate_k_gradients), emsize
        src = src.permute(1, 0, 2)
        # After: (batch_size / aggregate_k_gradients), BPTT, emsize

        hidden_states = self.mamba_backbone(src, inference_parameters=None)

        # Before: (batch_size / aggregate_k_gradients), BPTT, emsize
        hidden_states = hidden_states.permute(1, 0, 2)
        # After: BPTT, (batch_size / aggregate_k_gradients), emsize

        output = self.decoder(hidden_states)
        return output[single_eval_pos:]