# %%
import numpy as np
import kagglehub
from jax.sharding import Mesh
from hydra.core.global_hydra import GlobalHydra
import jax.numpy as jnp
import jax
from flax.training import checkpoints
from einops import rearrange, einsum
import sys
import os
from importlib import reload
from gemma import params as params_lib
from gemma import modules
from gemma import transformer as transformer_lib

# import jax_extra
# from train import Config, Model, training_step, State
# from input_loader import HuggingFaceDataParams, HuggingFaceDataLoader, TokenBatchParams, TokenBatch
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


kagglehub.login()
# %%
kagglehub.model_download("google/gemma-2/flax/gemma2-2b")

# %%
model_dir = "/Users/clankur/.cache/kagglehub/models"
gemma_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1"
checkpoint_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1/gemma2-2b"
# %%
batch_size = 1
seq_length = 5  # tokens.targets.shape[-1]

# %%
params = params_lib.load_and_format_params(checkpoint_dir)
# %%
reload(modules)
reload(transformer_lib)

gemma2_config = transformer_lib.TransformerConfig.gemma2_2b(1024)
gemma2_config = gemma2_config.from_params(params=params)
transformer = transformer_lib.Transformer(gemma2_config)
# %%

dummy_input = jnp.zeros((batch_size, seq_length)).astype(jnp.int32)
dummy_positions = jnp.arange(seq_length)[None, :]
dummy_attention_mask = jnp.ones((batch_size, seq_length, seq_length))
# %%
(logits, cache), intermediates = transformer.apply(
    {'params': params["transformer"]},
    dummy_input,
    dummy_positions,
    None,
    dummy_attention_mask,
    capture_intermediates=True

)
# %%
print(intermediates['intermediates'].keys())
print(intermediates['intermediates']['layer_0'].keys())
print(intermediates['intermediates']['layer_0']
      ["attn"].keys())


# %%


def flatten_intermediates(intermediates):
    flattened = {}
    num_layers = len(
        [key for key in intermediates['intermediates'] if key.startswith('layer_')])

    def extract_value(d):
        if '__call__' in d:
            return d['__call__'][0]
        for v in d.values():
            if isinstance(v, dict):
                return extract_value(v)
        return None

    main_keys = ['pre_attention_norm', 'post_attention_norm',
                 'pre_ffw_norm', 'mlp', 'post_ffw_norm']
    attn_keys = ['q_einsum', 'kv_einsum',
                 'attn_vec_einsum', 'roped_q', "roped_k", "scaled_q", "att_logits", "capped_logits"]

    for key in main_keys + attn_keys:
        layer_values = []
        for i in range(num_layers):
            layer_key = f'layer_{i}'
            if key in main_keys:
                value = extract_value(
                    intermediates['intermediates'][layer_key][key])
            else:  # attn keys
                value = extract_value(
                    intermediates['intermediates'][layer_key]['attn'][key])
            if value is not None:
                layer_values.append(value)
        if layer_values:
            flattened[key] = jnp.stack(layer_values)

    # Handle special cases
    flattened['tracked_embed'] = intermediates['intermediates']['tracked_embed']['__call__'][0]
    flattened['final_norm'] = intermediates['intermediates']['final_norm']['__call__'][0]
    flattened['tracked_unembed'] = intermediates['intermediates']['tracked_unembed']['__call__'][0]

    return flattened


# %%
intermediates = flatten_intermediates(intermediates)

# %%


class Hparams:
    d_model: int = gemma2_config.embed_dim
    n_q_per_kv: int = gemma2_config.num_heads // gemma2_config.num_kv_heads
    n_kv: int = gemma2_config.num_kv_heads
    d_head: int = gemma2_config.head_dim
    layers: int = gemma2_config.num_layers
    vocab: int = gemma2_config.num_embed
    d_ff: int = gemma2_config.hidden_dim
    window_size: int = gemma2_config.sliding_window_size
    attn_softcap: float = gemma2_config.attn_logits_soft_cap
    final_softcap: float = gemma2_config.final_logit_softcap
    rope_max_timescale: int = 10_000

# %%


def compare_tensors(tensor1: jax.Array, tensor2: jax.Array, tolerance: float = 1e-5) -> tuple[bool, bool]:
    if tensor1.shape != tensor2.shape:
        return False, False

    exact_match = jnp.array_equal(tensor1, tensor2)

    max_diff = jnp.max(jnp.abs(tensor1 - tensor2))
    approximate_match = max_diff <= tolerance

    return exact_match, approximate_match


def rms_norm(x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + 1e-06)


class RopeTable:
    def __init__(self, max_len: int, h: Hparams) -> None:
        head_dim = h.d_head
        position = jnp.arange(max_len, dtype=jnp.int32)
        fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        timescale = h.rope_max_timescale**fraction

        sinusoid_inp = jnp.float32(
            position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        self.sin = jnp.sin(sinusoid_inp)
        self.cos = jnp.cos(sinusoid_inp)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = rearrange(self.sin, rearrange_spec)
        cos = rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.concatenate([r1, r2], axis=-1).astype(x.dtype)


# %%
h = Hparams()
# %%
# weight init
weights = params['transformer']
rope_table = RopeTable(seq_length, h)
unembed = embed = weights['embedder']['input_embedding']
ln1 = weights["layer_0"]['pre_attention_norm']['scale']
w_q = weights["layer_0"]['attn']['q_einsum']['w']
w_q = rearrange(
    w_q,
    "(n_q_per_kv n_kv) d_model d_head -> d_model n_q_per_kv n_kv d_head",
    n_q_per_kv=h.n_q_per_kv,
    n_kv=h.n_kv,
)
w_kv = weights["layer_0"]['attn']['kv_einsum']['w']
w_kv = rearrange(
    w_kv,
    ("k_v n_kv M_dim H_dim -> k_v M_dim n_kv H_dim")
)
# %%
# forward pass
ids = dummy_input
x = embed[ids]
x *= jnp.sqrt(h.d_model)
nx = rms_norm(x) * (1.0 + ln1)
q = einsum(
    nx, w_q, "B Qlen M_dim, M_dim n_per_kv n_kv H_dim -> B Qlen n_kv n_per_kv H_dim")
k, v = einsum(
    nx, w_kv, "B Klen M_dim, k_v M_dim n_kv H_dim -> k_v B Klen n_kv H_dim"
)
# %%
q = rope_table.apply("L d -> 1 L 1 1 d", q)
k = rope_table.apply("L d -> 1 L 1 d", k)
# %%

intermediates['roped_q_'] = rearrange(
    intermediates['roped_q'], "layer b Qlen (n_q_per_kv n_kv) d_head -> layer b Qlen n_kv n_q_per_kv d_head",
    n_q_per_kv=h.n_q_per_kv,
    n_kv=h.n_kv,
)
print(compare_tensors(q, intermediates['roped_q_'][0]))
q.shape, intermediates['roped_q_'][0].shape
# %%
q_preatt_scalar = h.d_head ** -0.5
q_scaled = q * q_preatt_scalar

# %%
intermediates['scaled_q_'] = rearrange(
    intermediates['scaled_q'], "layer b Qlen (n_q_per_kv n_kv) d_head -> layer b Qlen n_kv n_q_per_kv d_head",
    n_q_per_kv=h.n_q_per_kv,
    n_kv=h.n_kv,
)
print(compare_tensors(q_scaled, intermediates['scaled_q_'][0]))

# logits = jnp.einsum('BTKGH,BSKH->BTKGS', q_scaled, k)
# logits = logits.reshape((b, t, n_kv * g, s))

# %%
intermediates['q'] = rearrange(
    intermediates['q_einsum'], "layer b Qlen (n_q_per_kv n_kv) d_head -> layer b Qlen n_q_per_kv n_kv d_head",
    n_q_per_kv=h.n_q_per_kv,
    n_kv=h.n_kv,
)
intermediates['k'] = intermediates['kv_einsum'][:, 0]
intermediates['v'] = intermediates['kv_einsum'][:, 1]

intermediates['att_logits'][0].shape

# %%
print(compare_tensors(x, intermediates["tracked_embed"]))
print(compare_tensors(nx, intermediates["pre_attention_norm"][0]))
print(compare_tensors(v, intermediates['v'][0]))
print(compare_tensors(k, intermediates['roped_k'][0]))
print(compare_tensors(logits, intermediates["att_logits"][0]))
# %%
intermediates["att_logits"][0]
# %%

print(compare_tensors(logits, intermediates["att_logits"][0]))
logits = rearrange(
    logits, "B Qlen Klen n_q_per_kv n_kv -> B Qlen (n_q_per_kv n_kv) Klen ")
#
# %%
logits - intermediates["att_logits"][0]


# %%
logits = einsum(
    intermediates['roped_q'][0],
    intermediates['roped_k'][0],
    "B Qlen n_kv n_q_per_kv D, B Klen n_kv D -> B Qlen Klen n_q_per_kv n_kv",
)

logits = rearrange(
    logits, "B Qlen Klen n_q_per_kv n_kv -> B Qlen (n_q_per_kv n_kv) Klen ")
# %%
logits - intermediates["att_logits"][0]

# %%
