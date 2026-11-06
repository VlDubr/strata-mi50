"""tools/dsv41/pack_glm5.py on a small made-up GGUF directory: the name map by layer kind, the experts split one entry each,
block 45 under nextn.*, and the checks that stop a pack with a hole.  No model file."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "dsv41"))
import pack_glm5 as P  # noqa: E402

F32, Q8, Q4 = 0, 8, 12
N_EXP = 2


def kv():
    a = P.ARCH
    return {
        "general.architecture": a, "%s.block_count" % a: 5, "%s.nextn_predict_layers" % a: 1,
        "%s.attention.head_count_kv" % a: [0, 1, 0, 1, 1], "%s.embedding_length" % a: 256, "%s.feed_forward_length" % a: 512,
        "%s.attention.head_count" % a: 2, "%s.attention.kv_lora_rank" % a: 64, "%s.attention.key_length_mla" % a: 32,
        "%s.attention.value_length_mla" % a: 32, "%s.rope.dimension_count" % a: 0, "%s.attention.q_lora_rank" % a: 64,
        "%s.expert_count" % a: N_EXP, "%s.expert_used_count" % a: 1, "%s.expert_feed_forward_length" % a: 256,
        "%s.expert_shared_feed_forward_length" % a: 256, "%s.leading_dense_block_count" % a: 1, "%s.expert_shared_count" % a: 1,
        "%s.expert_group_count" % a: 1, "%s.hyper_connection.count" % a: 4, "%s.hyper_connection.sinkhorn_iterations" % a: 20,
        "%s.hyper_connection.epsilon" % a: 1e-6, "%s.attention.indexer.head_count" % a: 2, "%s.attention.indexer.key_length" % a: 32,
        "%s.attention.indexer.top_k" % a: 8, "%s.attention.indexer.kpool" % a: 4, "%s.kda.head_dim" % a: 32,
        "%s.kda.gate_lower_bound" % a: -5.0, "%s.ssm.conv_kernel" % a: 4, "%s.attention.layer_norm_rms_epsilon" % a: 1e-5,
        "%s.attention.layer_norm_epsilon" % a: 1e-6, "%s.expert_weights_scale" % a: 2.5, "%s.swiglu_clamp_exp" % a: [10.0] * 5,
        "%s.swiglu_clamp_shexp" % a: [10.0] * 5, "%s.context_length" % a: 1024, "tokenizer.ggml.eos_token_id": 7,
        "tokenizer.ggml.eot_token_id": 8, "tokenizer.ggml.eom_token_id": 9,
    }


class FakeGguf:
    path = pathlib.Path("model-00001-of-00001.gguf")
    data_start = 100

    def __init__(self, drop=()):
        self.tensors = []
        off = 0

        def add(name, typ, dims):
            nonlocal off
            if name not in drop:
                self.tensors.append((name, typ, dims, off))
            off += 4096

        add("token_embd.weight", Q8, [256, 16])
        add("output_norm.weight", F32, [256])
        add("output.weight", Q8, [256, 16])
        kinds = [0, 1, 0, 1, 1]
        for il in range(5):
            b = "blk.%d." % il
            for k in ("attn_norm.weight", "ffn_norm.weight"):
                add(b + k, F32, [256])
            if il < 4:
                for k in ("hc_attn_fn.weight", "hc_ffn_fn.weight"):
                    add(b + k, Q8, [1024, 24])
                for k in ("hc_attn_base.weight", "hc_ffn_base.weight"):
                    add(b + k, F32, [24])
                for k in ("hc_attn_scale.weight", "hc_ffn_scale.weight"):
                    add(b + k, F32, [3])
            table = P.MLA if kinds[il] else P.KDA
            for k in table:
                add(b + k, F32, [32])
            if il == 0:
                for k in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                    add(b + k, Q8, [256, 512])
            else:
                add(b + "ffn_gate_inp.weight", F32, [256, N_EXP])
                add(b + "exp_probs_b.bias", F32, [N_EXP])
                for k in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
                    add(b + k, Q8, [256, 256])
                for k in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
                    add(b + k, Q4, [256, 256, N_EXP])
            if il == 4:
                for k in P.NEXTN:
                    add(b + k, F32, [256])


class Pack(unittest.TestCase):
    def setUp(self):
        self.kv = kv()
        self.tensors, self.files = P.collect([FakeGguf()], self.kv)
        self.names = {t[0] for t in self.tensors}
        self.geo = P.geometry(self.kv, 16, 2)

    def test_geometry(self):
        self.assertEqual(self.geo["num_hidden_layers"], 4)
        self.assertEqual(self.geo["mla_layers"], [1, 3])
        self.assertEqual(self.geo["eos_token_ids"], [7, 8, 9])
        self.assertEqual(self.geo["family"], P.ARCH)

    def test_output_matrix_goes_by_layer_kind(self):
        self.assertIn("layers.0.kda.wo.weight", self.names)
        self.assertIn("layers.1.attn.wo.weight", self.names)
        self.assertNotIn("layers.1.kda.wo.weight", self.names)

    def test_experts_split_one_entry_each(self):
        e = {t[0]: t for t in self.tensors}
        a, b = e["layers.1.ffn.experts.0.w1.weight"], e["layers.1.ffn.experts.1.w1.weight"]
        self.assertEqual(a[5], [256, 256])
        self.assertEqual(b[3] - a[3], a[4])

    def test_last_block_is_nextn(self):
        self.assertIn("nextn.eh_proj.weight", self.names)
        self.assertIn("nextn.ffn.experts.1.w2.weight", self.names)
        self.assertFalse(any(n.startswith("layers.4.") for n in self.names))

    def test_check_passes_a_whole_pack(self):
        P.check(self.tensors, self.geo)

    def test_check_stops_a_hole(self):
        t, _ = P.collect([FakeGguf(drop=("blk.2.ssm_a",))], self.kv)
        with self.assertRaises(SystemExit):
            P.check(t, self.geo)

    def test_unknown_tensor_stops_the_pack(self):
        g = FakeGguf()
        g.tensors.append(("blk.0.something_new.weight", F32, [32], 0))
        with self.assertRaises(SystemExit):
            P.collect([g], self.kv)

    def test_clamp_must_be_one_value(self):
        k = kv()
        k["%s.swiglu_clamp_exp" % P.ARCH] = [10.0, 9.0, 10.0, 10.0, 10.0]
        with self.assertRaises(SystemExit):
            P.geometry(k, 16, 2)


if __name__ == "__main__":
    unittest.main()
