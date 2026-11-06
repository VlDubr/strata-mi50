"""tools/dsv41/pack_glm5.py - the engine's pack for a GLM-5.3-Flash GGUF (architecture glm5-next), beside the model's files
(no weights copied).

    python tools/dsv41/pack_glm5.py --model ~/models/GLM-5.3-Flash-UD-Q4_K_XL/UD-Q4_K_XL

writes <model>/strata-glm5/ in pack_gguf.py's format (ds4.txt, tensors.txt, tokenizer/), with `family glm5-next` in ds4.txt.
Names follow the DeepSeek pack where the tensor has the same role; the rest are new:

  layers.N.attn.*     the MLA layers (attention.head_count_kv[N] == 1): wq_a, q_norm, wq_b, wkv, kv_norm, k_b, v_b, wo, and
                      attn.indexer.* (wq_b, wk, k_norm, weights_proj, comp_gate, comp_ape)
  layers.N.kda.*      the KDA layers (head_count_kv[N] == 0): wq, wk, wv, conv_q/k/v, f_a, f_b, dt_bias, a, beta, g_a, g_b,
                      norm, wo
  layers.N.ffn.dense.w1/w2/w3    the leading dense blocks
  nextn.*             the last block (nextn_predict_layers): the multi-token head, not part of the trunk, so
                      num_hidden_layers does not count it

A KDA layer and an MLA layer both call their output matrix attn_output in the GGUF, so the map depends on the layer's kind.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pack_gguf import TYPES, Gguf, short, write_tokenizer  # noqa: E402

ARCH = "glm5-next"

# GGUF name (blk.N. stripped) -> the pack's name under layers.N.
COMMON = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "ffn_gate.weight": "ffn.dense.w1.weight",
    "ffn_up.weight": "ffn.dense.w3.weight",
    "ffn_down.weight": "ffn.dense.w2.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_gate_exps.weight": "ffn.experts.w1",    # [n_exp][ff][dim]: one entry an expert
    "ffn_up_exps.weight": "ffn.experts.w3",
    "ffn_down_exps.weight": "ffn.experts.w2",    # [n_exp][dim][ff]
}
MLA = {
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv_a_mqa.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_k_b.weight": "attn.k_b.weight",        # [heads][latent][qk head]
    "attn_v_b.weight": "attn.v_b.weight",        # [heads][v head][latent]
    "attn_output.weight": "attn.wo.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "attn.indexer.k_norm.bias",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "indexer_compressor_gate.weight": "attn.indexer.comp_gate.weight",
    "indexer_compressor_ape.weight": "attn.indexer.comp_ape.weight",
}
KDA = {
    "attn_q.weight": "kda.wq.weight",
    "attn_k.weight": "kda.wk.weight",
    "attn_v.weight": "kda.wv.weight",
    "attn_output.weight": "kda.wo.weight",
    "ssm_conv1d_q.weight": "kda.conv_q.weight",
    "ssm_conv1d_k.weight": "kda.conv_k.weight",
    "ssm_conv1d_v.weight": "kda.conv_v.weight",
    "ssm_f_a.weight": "kda.f_a.weight",
    "ssm_f_b.weight": "kda.f_b.weight",
    "ssm_dt.bias": "kda.dt_bias",
    "ssm_a": "kda.a",
    "ssm_beta.weight": "kda.beta.weight",
    "ssm_g_a.weight": "kda.g_a.weight",
    "ssm_g_b.weight": "kda.g_b.weight",
    "ssm_norm.weight": "kda.norm.weight",
}
NEXTN = {
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.shared_head_norm.weight": "shared_head_norm.weight",
}
GLOBAL = {"token_embd.weight": "embed.weight", "output_norm.weight": "norm.weight", "output.weight": "head.weight"}
# every role a trunk layer of each kind must end up with (the experts are checked apart: all n_exp of each role)
HC = ("attn_norm.weight", "ffn_norm.weight", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base",
      "hc_ffn_scale")
DENSE = ("ffn.dense.w1.weight", "ffn.dense.w2.weight", "ffn.dense.w3.weight")
MOE = ("ffn.gate.weight", "ffn.gate.bias", "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
       "ffn.shared_experts.w3.weight")


def geometry(kv: dict, vocab: int, kda_heads: int) -> dict:
    """ds4.txt's keys: the DeepSeek pack's names where the quantity is the same, new ones for what only GLM has."""
    def g(k, d=None):
        return kv.get("%s.%s" % (ARCH, k), d)

    nextn = int(g("nextn_predict_layers", 0))
    nl = int(g("block_count")) - nextn
    kinds = list(g("attention.head_count_kv"))
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        vals = list(g(key))
        if any(abs(v - vals[0]) > 1e-6 for v in vals):
            raise SystemExit("%s differs between layers: the engine takes one swiglu_limit" % key)
    if abs(g("swiglu_clamp_exp")[0] - g("swiglu_clamp_shexp")[0]) > 1e-6:
        raise SystemExit("swiglu_clamp_exp and swiglu_clamp_shexp differ: the engine takes one swiglu_limit")
    if int(g("rope.dimension_count")) != 0:
        raise SystemExit("rope.dimension_count is not 0: the GLM path has no RoPE")
    if int(g("expert_shared_count")) != 1 or int(g("expert_group_count")) != 1:
        raise SystemExit("the GLM path takes one shared expert and one expert group")
    eos = [int(kv[k]) for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id")
           if k in kv]
    return {
        "family": ARCH,
        "vocab_size": vocab,
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": nl,
        "nextn_layers": nextn,
        "mla_layers": [i for i in range(nl) if kinds[i] == 1],
        "num_attention_heads": g("attention.head_count"),
        "head_dim": g("attention.kv_lora_rank"),          # the cached latent: the width sparse attention reads
        "mla_qk_head_dim": g("attention.key_length_mla"),
        "mla_v_head_dim": g("attention.value_length_mla"),
        "qk_rope_head_dim": 0,
        "q_lora_rank": g("attention.q_lora_rank"),
        "n_routed_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "moe_intermediate_size": g("expert_feed_forward_length"),
        "shared_intermediate_size": g("expert_shared_feed_forward_length"),
        "dense_intermediate_size": g("feed_forward_length"),
        "dense_layers": g("leading_dense_block_count"),
        "hc_mult": g("hyper_connection.count"),
        "hc_sinkhorn_iters": g("hyper_connection.sinkhorn_iterations"),
        "hc_eps": g("hyper_connection.epsilon"),
        "index_n_heads": g("attention.indexer.head_count"),
        "index_head_dim": g("attention.indexer.key_length"),
        "index_topk": g("attention.indexer.top_k"),
        "index_kpool": g("attention.indexer.kpool"),
        "kda_heads": kda_heads,
        "kda_head_dim": g("kda.head_dim"),
        "kda_gate_lower_bound": g("kda.gate_lower_bound"),
        "kda_conv_kernel": g("ssm.conv_kernel"),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
        "layer_norm_eps": g("attention.layer_norm_epsilon"),
        "routed_scaling_factor": g("expert_weights_scale"),
        "swiglu_limit": g("swiglu_clamp_exp")[0],
        "max_position_embeddings": g("context_length"),
        "eos_token_id": eos[0],
        "eos_token_ids": eos,
    }


def collect(gs: list, kv: dict):
    """(tensors, file names): every tensor of the shards under the pack's name; an unknown name stops the pack."""
    kinds = list(kv["%s.attention.head_count_kv" % ARCH])
    nl = int(kv["%s.block_count" % ARCH]) - int(kv.get("%s.nextn_predict_layers" % ARCH, 0))
    tensors, unknown = [], []
    for fi, g in enumerate(gs):
        for name, typ, dims, off in g.tensors:
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (g.path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            if dims[0] % per:
                raise SystemExit("%s: %s rows of %d are not whole %s blocks" % (g.path.name, name, dims[0], tname))
            nbytes = nel // per * bpb
            shape = list(reversed(dims))     # ggml ne0 innermost -> torch order, outermost first
            base = g.data_start + off
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            if m:
                il, part = int(m.group(1)), m.group(2)
                table = {**COMMON, **(MLA if kinds[il] == 1 else KDA)}
                prefix = "layers.%d." % il if il < nl else "nextn."
                if il >= nl and part in NEXTN:
                    ck = prefix + NEXTN[part]
                elif part in table:
                    ck = prefix + table[part]
                else:
                    unknown.append(name)
                    continue
            elif name in GLOBAL:
                ck = GLOBAL[name]
            else:
                unknown.append(name)
                continue
            if ".ffn.experts." in ck:
                # [n_exp][out][in] -> an entry an expert: layers.N.ffn.experts.E.w1.weight
                n, part_b = shape[0], nbytes // shape[0]
                pre, role = ck.rsplit(".", 1)
                for i in range(n):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, role), tname, fi, base + i * part_b, part_b, shape[1:]))
                continue
            tensors.append((ck, tname, fi, base, nbytes, shape))
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    return tensors, [g.path.name for g in gs]


def check(tensors: list, geo: dict):
    """Every role of every layer is there, once: a pack with a hole must stop here, not at load."""
    names = [t[0] for t in tensors]
    have = set(names)
    if len(have) != len(names):
        raise SystemExit("the pack has duplicate tensor names")
    missing = []
    mla = set(geo["mla_layers"])
    for il in range(geo["num_hidden_layers"]):
        want = list(HC)
        want += list(MLA[k] for k in MLA) if il in mla else list(KDA[k] for k in KDA)
        if il < geo["dense_layers"]:
            want += DENSE
        else:
            want += MOE
            want += ["ffn.experts.%d.%s.weight" % (e, r) for e in range(geo["n_routed_experts"]) for r in ("w1", "w2", "w3")]
        missing += ["layers.%d.%s" % (il, w) for w in want if "layers.%d.%s" % (il, w) not in have]
    missing += [n for n in GLOBAL.values() if n not in have]
    if missing:
        raise SystemExit("the pack lacks %d tensors: %s" % (len(missing), ", ".join(missing[:12])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the folder of the GGUF's shards")
    ap.add_argument("--out", help="the pack's folder (default <model>/strata-glm5)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    paths = sorted(model.glob("*.gguf"))
    paths = [p for p in paths if not p.name.startswith("mmproj")]
    if not paths:
        raise SystemExit("%s: no .gguf files" % model)
    gs = [Gguf(p) for p in paths]
    kv = gs[0].kv
    if kv.get("general.architecture") != ARCH:
        raise SystemExit("%s: architecture %r, not %s" % (paths[0].name, kv.get("general.architecture"), ARCH))
    count = int(kv.get("split.count", 1))
    if count != len(gs):
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(gs), count))
    why = next((w for w in map(short, gs) if w), None)
    if why:
        raise SystemExit("not whole: %s" % why)
    seen = sum(len(g.tensors) for g in gs)
    if "split.tensors.count" in kv and seen != int(kv["split.tensors.count"]):
        raise SystemExit("%d tensors in the shards, the header says %s" % (seen, kv["split.tensors.count"]))

    tensors, files = collect(gs, kv)
    by_name = {t[0]: t for t in tensors}
    vocab = by_name["embed.weight"][5][0]
    wq = next(t for t in tensors if t[0].endswith(".kda.wq.weight"))
    kda_heads = wq[5][0] // int(kv["%s.kda.head_dim" % ARCH])
    geo = geometry(kv, vocab, kda_heads)
    check(tensors, geo)

    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model / "strata-glm5"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ds4.txt").open("w") as f:
        for k, v in geo.items():
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        f.write("format gguf\n")
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    template = kv.get("tokenizer.chat_template")
    if not template:
        raise SystemExit("%s: the GGUF has no tokenizer.chat_template" % model)
    write_tokenizer(out / "tokenizer", kv, template)
    by = {}
    for t in tensors:
        by[t[1]] = by.get(t[1], 0) + t[4]
    print("%s: %d tensors in %d files (%s) -> %s" % (model.name, len(tensors), len(files),
                                                     ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
