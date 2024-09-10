import jax
from einops import rearrange
from jax import jnp
from gemma import params as params_lib
from gemma import transformer as transformer_lib
import kagglehub

kagglehub.login()
kagglehub.model_download("google/gemma-2/flax/gemma2-2b")

# %%
model_dir = "/Users/clankur/.cache/kagglehub/models"
gemma_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1"
checkpoint_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1/gemma2-2b"
batch_size = 1
L = 5
params = params_lib.load_and_format_params(checkpoint_dir)

gemma2_config = transformer_lib.TransformerConfig.gemma2_2b(1024)
gemma2_config = gemma2_config.from_params(params=params)
transformer = transformer_lib.Transformer(gemma2_config)
# %%
dummy_input = jnp.zeros((batch_size, L)).astype(jnp.int32)
dummy_positions = jnp.arange(L)[None, :]
dummy_attention_mask = jnp.tril(
    jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0
)


def rms_norm(x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + 1e-06)


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


h = Hparams()


class Model:
    def __init__(
        self,
        embed,
        w_q, w_kv, w_o,
        w_gate, w_up, w_down,
        ln1, ln2,
        post_attn_ln,
        post_ffn_ln,
        final_layer_norm
    ):
        self.embed = embed
        self.w_q = w_q
        self.w_kv = w_kv
        self.w_o = w_o
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.ln1 = ln1
        self.ln2 = ln2
        self.post_attn_ln = post_attn_ln
        self.post_ffn_ln = post_ffn_ln
        self.final_layer_norm = final_layer_norm

    @classmethod
    def load_weights(cls, params):
        num_layers = h.layers

        embed = params['transformer']['embedder']['input_embedding']
        final_layer_norm = params['transformer']['final_norm']['scale']

        w_q = []
        w_kv = []
        w_o = []
        w_gate = []
        w_up = []
        w_down = []
        ln1 = []
        ln2 = []
        post_attn_ln = []
        post_ffn_ln = []

        for i in range(num_layers):
            layer_key = f'layer_{i}'
            layer = params['transformer'][layer_key]

            ln1.append(layer['pre_attention_norm']['scale'])
            ln2.append(layer['pre_ffw_norm']['scale'])

            w_q_layer = layer['attn']['q_einsum']['w']
            w_q_layer = rearrange(w_q_layer, "(n_kv n_q_per_kv) d_model d_head -> d_model n_kv n_q_per_kv d_head",
                                  n_q_per_kv=h.n_q_per_kv, n_kv=h.n_kv)
            w_q.append(w_q_layer)

            w_kv_layer = layer['attn']['kv_einsum']['w']
            w_kv_layer = rearrange(
                w_kv_layer, "k_v n_kv M_dim H_dim -> k_v M_dim n_kv H_dim")
            w_kv.append(w_kv_layer)

            w_o.append(layer['attn']['attn_vec_einsum']['w'])

            post_attn_ln.append(layer['post_attention_norm']['scale'])
            w_gate.append(layer['mlp']['gating_einsum'][0])
            w_up.append(layer['mlp']['gating_einsum'][1])
            w_down_layer = rearrange(layer['mlp']['linear'], "F M -> M F")
            w_down.append(w_down_layer)
            post_ffn_ln.append(layer['post_ffw_norm']['scale'])

        # Stack layer-specific tensors
        w_q = jnp.stack(w_q)
        w_kv = jnp.stack(w_kv)
        w_o = jnp.stack(w_o)
        w_gate = jnp.stack(w_gate)
        w_up = jnp.stack(w_up)
        w_down = jnp.stack(w_down)
        ln1 = jnp.stack(ln1)
        ln2 = jnp.stack(ln2)
        post_attn_ln = jnp.stack(post_attn_ln)
        post_ffn_ln = jnp.stack(post_ffn_ln)

        return cls(
            embed,
            w_q,
            w_kv,
            w_o,
            w_gate,
            w_up,
            w_down,
            ln1,
            ln2,
            post_attn_ln,
            post_ffn_ln,
            final_layer_norm
        )

    def forward_pass(self, ids):
        rope_table = RopeTable(L, h)
        causal_mask = jnp.tril(
            jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0
        )[..., jnp.newaxis, jnp.newaxis, :]
        local_mask = jnp.triu(
            jnp.ones((batch_size, L, L), dtype=jnp.bool_), 1 - h.window_size
        )[..., jnp.newaxis, jnp.newaxis, :]

        def loop_body(carry, layer_params):
            x, use_local_window_attn = carry
            (w_q,
             w_kv,
             w_o,
             w_gate,
             w_up,
             w_down,
             ln1,
             ln2,
             post_attn_ln,
             post_ffn_ln) = layer_params

            nx = rms_norm(x) * (1.0 + ln1)
            q = jnp.einsum('btd,dnqh->btnqh', nx, w_q)
            k, v = jnp.einsum('bld,kdnh->kblnh', nx, w_kv)

            q = rope_table.apply("L d -> 1 L 1 1 d", q)
            k = rope_table.apply("L d -> 1 L 1 d", k)

            q_scaled = q * (h.d_head ** -0.5)
            logits = jnp.einsum('btnqh,blnh->btnql', q_scaled, k)
            logits = jnp.tanh(logits / h.attn_softcap) * h.attn_softcap

            attn_mask = jax.lax.select(
                use_local_window_attn,
                jnp.logical_and(causal_mask, local_mask),
                causal_mask
            )

            logits = jnp.where(attn_mask, logits, -2.3819763e38)

            probs = jax.nn.softmax(logits, axis=-1).astype(logits.dtype)

            encoded = jnp.einsum('btnql,blnh->btnqh', probs, v)
            encoded = rearrange(encoded, "b l n q h -> b l (n q) h")

            attn_out = jnp.einsum('blnh,nhd->bld', encoded, w_o)
            attn_out = rms_norm(attn_out) * (1.0 + post_attn_ln)
            x += attn_out

            # FFN
            nx = rms_norm(x) * (1.0 + ln2)
            gate_proj = jnp.einsum('bld,df->blf', nx, w_gate)
            up_proj = jnp.einsum('bld,df->blf', nx, w_up)
            y = jax.nn.gelu(gate_proj) * up_proj
            ffn_out = jnp.einsum('blf,df->bld', y, w_down)
            ffn_out = rms_norm(ffn_out) * (1.0 + post_ffn_ln)
            x = x + ffn_out

            return (x, ~use_local_window_attn), ()

        x = self.embed[ids]
        use_local_window_attn = False
        x *= jnp.sqrt(h.d_model)

        for i in range(h.layers):
            layer_weights = [
                self.w_q[i],
                self.w_kv[i],
                self.w_o[i],
                self.w_gate[i],
                self.w_up[i],
                self.w_down[i],
                self.ln1[i],
                self.ln2[i],
                self.post_attn_ln[i],
                self.post_ffn_ln[i]
            ]
            (x, use_local_window_attn), _ = loop_body(
                (x, use_local_window_attn),
                layer_weights
            )
        x = rms_norm(x) * (1.0 + self.final_layer_norm)
        logits = jnp.einsum("blm,vm->blv", x, self.embed)
        logits = jnp.tanh(logits / h.final_softcap) * h.final_softcap
        return logits


# %%
model = Model.load_weights(params)

# %%
model.forward_pass(dummy_input)
# %%
