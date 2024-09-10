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

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


kagglehub.login()
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
dummy_attention_mask = jnp.tril(
    jnp.ones((batch_size, seq_length, seq_length), dtype=jnp.bool_), 0
)

(logits, cache), intermediates = transformer.apply(
    {'params': params["transformer"]},
    dummy_input,
    dummy_positions,
    None,
    dummy_attention_mask,
    capture_intermediates=True

)


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
                 'pre_ffw_norm', 'mlp', 'post_ffw_norm', "final_output"]
    attn_keys = ['q_einsum', 'kv_einsum', "reshaped_scaled_q",
                 'attn_vec_einsum', 'roped_q', "roped_k", "scaled_q",
                 "att_logits", "capped_logits", "att_wei", "a_out_premix",
                 "a_out", "masked_logits"
                 ]

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
    flattened['final_softcap'] = intermediates['intermediates']['final_softcap']['__call__'][0]
    flattened['tracked_unembed'] = intermediates['intermediates']['tracked_unembed']['__call__'][0]

    return flattened


intermediates = flatten_intermediates(intermediates)


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


def flatten_params_to_tensors(params, h):
    # Use h.n_layers instead of counting
    num_layers = h.layers

    # Process embeddings
    embed = params['transformer']['embedder']['input_embedding']

    final_layer_norm = params['transformer']['final_norm']['scale']
    # Initialize lists to hold layer-specific tensors
    pre_attention_norms = []
    pre_ffw_norms = []
    attn_qs = []
    attn_kvs = []
    attn_os = []
    post_attention_norms = []
    mlp_gates = []
    mlp_ups = []
    mlp_downs = []
    post_ffw_norms = []

    # Process each layer
    for i in range(num_layers):
        layer_key = f'layer_{i}'
        layer = params['transformer'][layer_key]

        # Attention related tensors
        pre_attention_norms.append(layer['pre_attention_norm']['scale'])
        pre_ffw_norms.append(layer['pre_ffw_norm']['scale'])

        w_q = layer['attn']['q_einsum']['w']
        w_q = rearrange(w_q, "(n_kv n_q_per_kv) d_model d_head -> d_model n_kv n_q_per_kv d_head",
                        n_q_per_kv=h.n_q_per_kv, n_kv=h.n_kv)
        attn_qs.append(w_q)

        w_kv = layer['attn']['kv_einsum']['w']
        w_kv = rearrange(w_kv, "k_v n_kv M_dim H_dim -> k_v M_dim n_kv H_dim")
        attn_kvs.append(w_kv)

        w_o = layer['attn']['attn_vec_einsum']['w']
        attn_os.append(w_o)

        # MLP related tensors
        post_attention_norms.append(layer['post_attention_norm']['scale'])
        mlp_gates.append(layer['mlp']['gating_einsum'][0])
        mlp_ups.append(layer['mlp']['gating_einsum'][1])
        w_down = rearrange(layer['mlp']['linear'], "F M -> M F")
        mlp_downs.append(w_down)
        post_ffw_norms.append(layer['post_ffw_norm']['scale'])

    # Stack layer-specific tensors
    pre_attention_norm = jnp.stack(pre_attention_norms)
    pre_ffw_norm = jnp.stack(pre_ffw_norms)
    attn_q = jnp.stack(attn_qs)
    attn_kv = jnp.stack(attn_kvs)
    attn_o = jnp.stack(attn_os)
    post_attention_norm = jnp.stack(post_attention_norms)
    mlp_gate = jnp.stack(mlp_gates)
    mlp_up = jnp.stack(mlp_ups)
    mlp_down = jnp.stack(mlp_downs)
    post_ffw_norm = jnp.stack(post_ffw_norms)

    return (
        embed,
        pre_attention_norm,
        pre_ffw_norm,
        attn_q,
        attn_kv,
        attn_o,
        post_attention_norm,
        mlp_gate,
        mlp_up,
        mlp_down,
        post_ffw_norm,
        final_layer_norm
    )


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
L = seq_length
h = Hparams()
K_MASK = -2.3819763e38
rope_table = RopeTable(seq_length, h)
use_local_window_attn = False
causal_mask = jnp.tril(
    jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0
)[..., jnp.newaxis, jnp.newaxis, :]
local_mask = jnp.triu(
    jnp.ones((batch_size, L, L), dtype=jnp.bool_), 1 - h.window_size
)[..., jnp.newaxis, jnp.newaxis, :]

# %%

# %%
(embed,
 m_ln1,
 m_ln2,
 m_w_q,
 m_w_kv,
 m_w_o,
 m_post_attn_ln,
 m_w_gate,
 m_w_up,
 m_w_down,
 m_post_ffn_ln,
 m_final_layer_norm) = flatten_params_to_tensors(params, h)

# %%
i = 0
ids = dummy_input
x = embed[ids]
x *= jnp.sqrt(h.d_model)
print(compare_tensors(x, intermediates['tracked_embed']))


def loop_body(carry, layer_weights):
    w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2, post_attn_ln, post_ffn_ln = layer_weights

    (x, use_local_window_attn, i) = carry

    jax.debug.print('layer {i} \n', i=i)
    print('initial carry dtype \n', x.dtype)

    nx = rms_norm(x) * (1.0 + ln1)
    jax.debug.print("normed x alignment={b}", b=compare_tensors(
        nx, intermediates['pre_attention_norm'][i]))

    # realigning
    # nx = intermediates['pre_attention_norm'][i]

    q = einsum(
        nx, w_q, "B Qlen d_model, d_model n_kv n_q_per_kv d_head -> B Qlen n_kv n_q_per_kv d_head"
    ).astype(x)
    k, v = einsum(
        nx, w_kv, "B Klen d_model, k_v d_model n_kv d_head -> k_v B Klen n_kv d_head"
    ).astype(x)

    q = rope_table.apply("L d -> 1 L 1 1 d", q)
    k = rope_table.apply("L d -> 1 L 1 d", k)
    q_preatt_scalar = h.d_head ** -0.5
    q_scaled = q * q_preatt_scalar

    jax.debug.print("roped_q alignment = {b}", b=compare_tensors(
        q_scaled, intermediates['reshaped_scaled_q'][i]))
    jax.debug.print("roped_k alignment = {b}", b=compare_tensors(
        k, intermediates['roped_k'][i]))

    logits = einsum(
        q_scaled, k, 'B Qlen n_kv n_q_per_kv d_head, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv Klen')
    logits = jnp.tanh(logits / h.attn_softcap) * h.attn_softcap
    logits_test = rearrange(
        logits, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen")
    jax.debug.print("capped logits alignment={b}", b=compare_tensors(
        logits_test, intermediates['capped_logits'][i]))

    attn_mask = jax.lax.select(
        use_local_window_attn,
        jnp.logical_and(causal_mask, local_mask),
        causal_mask,
    )
    logits = jnp.where(attn_mask, logits, -2.3819763e38)
    logits_test = rearrange(
        logits, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen"
    )
    jax.debug.print("masked logits alignment={b}", b=compare_tensors(
        logits_test, intermediates['masked_logits'][i]))

    probs = (jax.nn.softmax(logits, axis=-1).astype(x.dtype))
    probs_test = rearrange(
        probs, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen"
    )
    jax.debug.print("att wei alignment={b}", b=compare_tensors(
        probs_test, intermediates['att_wei'][i]))

    encoded = einsum(
        probs, v, "B Qlen n_kv n_q_per_kv Klen, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv d_head"
    )

    encoded = rearrange(
        encoded, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
    )

    # jax.debug.print("attn_out before MHA mix aligned: {b}", b=compare_tensors(
    #     encoded, intermediates['a_out_premix'][i]))

    # realigning
    encoded = intermediates['a_out_premix'][i]

    # for some reason: mixing mha is wrong here...
    # attn_out = einsum(
    #     encoded, w_o, "B Qlen n_head d_head, n_head d_head d_model  -> B Qlen d_model"
    # )
    # but correct here:
    attn_out = jnp.einsum('BTNH,NHD->BTD', encoded, w_o)

    jax.debug.print("attn_out after w_o alignment: {b}", b=compare_tensors(
        attn_out, intermediates['a_out'][i]))

    # realigning
    # attn_out = intermediates['a_out'][i]

    attn_out = rms_norm(attn_out) * (1.0 + post_attn_ln)
    jax.debug.print("post attn norm alignment = {b}", b=compare_tensors(
        attn_out, intermediates['post_attention_norm'][i]))

    # realigning
    attn_out = intermediates['post_attention_norm'][i]

    x += attn_out
    nx = rms_norm(x) * (1.0 + ln2)
    jax.debug.print("pre ffw norm alignment = {b}", b=compare_tensors(
        nx, intermediates['pre_ffw_norm'][i]))

    # realigning
    # nx = intermediates['pre_ffw_norm'][i]

    gate_proj = einsum(nx, w_gate, "B L M, M F -> B L F")
    up_proj = einsum(nx, w_up, "B L M, M F -> B L F")
    y = jax.nn.gelu(gate_proj) * up_proj
    ffn_out = einsum(y, w_down, "B L F, M F -> B L M")
    jax.debug.print("ffn_out alignment = {b}", b=compare_tensors(
        ffn_out, intermediates['mlp'][i]))

    # realigning
    # ffn_out = intermediates['mlp'][i]
    ffn_out = rms_norm(ffn_out) * (1.0 + post_ffn_ln)
    jax.debug.print("post_ffw_norm alignment = {b}", b=(compare_tensors(
        ffn_out, intermediates['post_ffw_norm'][i])))

    # realigning
    ffn_out = intermediates['post_ffw_norm'][i]
    x += ffn_out

    print('final carry dtype \n', x.dtype)

    return (jnp.bfloat16(x), ~use_local_window_attn, i+1), ()


for i in range(h.layers):
    layer_weights = [
        m_w_q[i],
        m_w_kv[i],
        m_w_o[i],
        m_w_gate[i],
        m_w_up[i],
        m_w_down[i],
        m_ln1[i],
        m_ln2[i],
        m_post_attn_ln[i],
        m_post_ffn_ln[i]
    ]
    (x, use_local_window_attn, i), _ = loop_body(
        (x, use_local_window_attn, i), layer_weights)

x = rms_norm(x) * (1.0 + m_final_layer_norm)
print(compare_tensors(
    x, intermediates['final_norm']))
logits = einsum(
    x, embed, "B L M, V M ->B L V"
)
print(compare_tensors(
    logits, intermediates['tracked_unembed']))
logits = jnp.tanh(logits / h.final_softcap) * h.final_softcap
print(compare_tensors(
    logits, intermediates['final_softcap']))


# %%
i = 0
ids = dummy_input
x = embed[ids]
x *= jnp.sqrt(h.d_model)
(x, _, _), () = jax.lax.scan(
    loop_body,
    (x, False, i),
    (
        m_w_q,
        m_w_kv,
        m_w_o,
        m_w_gate,
        m_w_up,
        m_w_down,
        m_ln1,
        m_ln2,
        m_post_attn_ln,
        m_post_ffn_ln
    ),
)
x = rms_norm(x) * (1.0 + m_final_layer_norm)
print(compare_tensors(
    x, intermediates['final_norm']))
logits = einsum(
    x, embed, "B L M, V M ->B L V"
)
print(compare_tensors(
    logits, intermediates['tracked_unembed']))
logits = jnp.tanh(logits / h.final_softcap) * h.final_softcap
print(compare_tensors(
    logits, intermediates['final_softcap']))


# %%
