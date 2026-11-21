import json
import math

import pytest

from model import architecture as arch
from src.data.taxonomy import LABELS

N_TYPES = len(arch.TYPE_LABELS)

# ---- labels and targets (no torch)


def test_type_labels_follow_the_taxonomy():
    assert arch.TYPE_LABELS == LABELS[1:] and "benign" not in arch.TYPE_LABELS
    assert N_TYPES == 9
    assert [arch.TYPE2ID[label] + 1 for label in arch.TYPE_LABELS] == list(range(1, len(LABELS)))


def test_module_imports_without_torch_symbols():
    assert callable(arch.build_classifier) and callable(arch.build_two_head)


def test_type_target_one_hot_without_probs():
    t = arch.type_target_from_record("prompt_leaking")
    assert t is not None and sum(t) == 1.0 and t[arch.TYPE2ID["prompt_leaking"]] == 1.0
    assert arch.type_target_from_record("prompt_leaking", {}) == t


def test_type_target_uses_and_renormalizes_probs():
    probs = {"prompt_leaking": 0.6, "instruction_override": 0.2, "other_injection": 0.0}
    t = arch.type_target_from_record("prompt_leaking", probs)
    assert sum(t) == pytest.approx(1.0)
    assert t[arch.TYPE2ID["prompt_leaking"]] == pytest.approx(0.75)
    assert t[arch.TYPE2ID["instruction_override"]] == pytest.approx(0.25)
    assert t[arch.TYPE2ID["context_manipulation"]] == 0.0


def test_type_target_falls_back_to_one_hot_on_bad_probs():
    for bad in ({"prompt_leaking": 0.0}, {"prompt_leaking": -1.0}, {"x": 1.0}):
        t = arch.type_target_from_record("prompt_leaking", bad)
        assert t[arch.TYPE2ID["prompt_leaking"]] == 1.0 and sum(t) == 1.0


@pytest.mark.parametrize("label", ["benign", "unlabeled", None, "made_up"])
def test_type_target_is_none_for_rows_the_type_head_must_skip(label):
    assert arch.type_target_from_record(label, {"prompt_leaking": 1.0}) is None


def test_estimated_size():
    assert arch.estimated_size_mb(54_600_000, 1) == pytest.approx(54.6)
    assert arch.estimated_size_mb(54_600_000, 4) == pytest.approx(218.4)


# ---- everything below needs torch + transformers

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_config(**over):
    cfg = transformers.ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        global_attn_every_n_layers=2,
        local_attention=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        classifier_dropout=0.0,
        **over,
    )
    return arch.configure(cfg)


def _tiny_model(**over):
    torch.manual_seed(0)
    return arch.HavelForInjectionDetection(_tiny_config(**over)).eval()


def _batch(b=4, n=10):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 64, (b, n), generator=g)
    mask = torch.ones(b, n, dtype=torch.long)
    mask[1, 6:] = 0  # one padded row
    ids[1, 6:] = 0
    return ids, mask


def test_forward_shapes():
    model = _tiny_model()
    ids, mask = _batch()
    out = model(input_ids=ids, attention_mask=mask)
    assert out.binary_logit.shape == (4,)
    assert out.type_logits.shape == (4, N_TYPES)


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_pooling_modes_run(pooling):
    model = arch.HavelForInjectionDetection(arch.configure(_tiny_config(), pooling)).eval()
    ids, mask = _batch()
    assert torch.isfinite(model(input_ids=ids, attention_mask=mask).binary_logit).all()


def test_padding_does_not_change_a_rows_output():
    model = _tiny_model()
    ids, mask = _batch()
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=mask)
        short = model(input_ids=ids[1:2, :6], attention_mask=mask[1:2, :6])
    assert torch.allclose(full.binary_logit[1], short.binary_logit[0], atol=1e-4)
    assert torch.allclose(full.type_logits[1], short.type_logits[0], atol=1e-4)


def test_wrong_number_of_types_is_rejected():
    cfg = _tiny_config()
    cfg.num_attack_types = 5
    with pytest.raises(ValueError, match="attack types"):
        arch.HavelForInjectionDetection(cfg)


def test_save_and_load_round_trip(tmp_path):
    model = _tiny_model()
    ids, mask = _batch()
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["num_attack_types"] == N_TYPES and saved["type_labels"] == list(arch.TYPE_LABELS)
    assert saved["architectures"] == ["HavelForInjectionDetection"]
    loaded = arch.HavelForInjectionDetection.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        a, b = model(input_ids=ids, attention_mask=mask), loaded(input_ids=ids, attention_mask=mask)
    assert torch.allclose(a.binary_logit, b.binary_logit, atol=1e-6)
    assert torch.allclose(a.type_logits, b.type_logits, atol=1e-6)


def test_heads_are_initialised_not_zero():
    model = _tiny_model()
    assert model.binary_head.weight.abs().sum() > 0 and model.type_head.weight.abs().sum() > 0
    assert model.binary_head.bias.abs().sum() == 0


def test_mmbert_small_with_32k_vocab_matches_the_documented_size():
    # mmBERT-small configuration (checked on Hugging Face), vocabulary pruned to 32,000.
    cfg = arch.configure(
        transformers.ModernBertConfig(
            vocab_size=32000,
            hidden_size=384,
            num_hidden_layers=22,
            intermediate_size=1152,
            num_attention_heads=6,
            global_attn_every_n_layers=3,
            local_attention=128,
            max_position_embeddings=8192,
            norm_bias=False,
            classifier_bias=False,
            pad_token_id=0,  # mmBERT-small's special-token ids; they must stay inside the vocabulary
            bos_token_id=2,
            eos_token_id=1,
            cls_token_id=1,
            sep_token_id=1,
        )
    )
    with torch.device("meta"):
        model = arch.HavelForInjectionDetection(cfg)
    counts = arch.parameter_counts(model)
    assert counts["embeddings"] == 32000 * 384
    assert counts["encoder"] == pytest.approx(42.19e6, abs=0.05e6)
    assert counts["shared_projection"] == 384 * 384 + 384
    assert counts["binary_head"] == 385 and counts["type_head"] == 384 * 9 + 9
    assert counts["total"] == pytest.approx(54.63e6, abs=0.05e6)
    assert arch.estimated_size_mb(counts["total"], 1) < 256


# ---- loss


def _logits(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, generator=g), torch.randn(b, N_TYPES, generator=g)


def test_loss_matches_a_manual_computation():
    bl, tl = _logits()
    y = torch.tensor([1, 0, 1, 1])
    targets = torch.zeros(4, N_TYPES)
    targets[0, 2] = 1.0
    targets[2] = torch.full((N_TYPES,), 1.0 / N_TYPES)
    targets[3, 0], targets[3, 1] = 0.5, 0.5
    mask = torch.tensor([True, False, True, True])
    total, parts = arch.two_head_loss(bl, tl, y, targets, mask, type_weight=0.5)

    p = torch.sigmoid(bl)
    bin_manual = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
    log_q = torch.log_softmax(tl, -1)
    rows = [-(targets[i] * log_q[i]).sum() for i in (0, 2, 3)]
    type_manual = torch.stack(rows).mean()
    assert parts["binary"].item() == pytest.approx(bin_manual.item(), abs=1e-5)
    assert parts["type"].item() == pytest.approx(type_manual.item(), abs=1e-5)
    assert total.item() == pytest.approx(bin_manual.item() + 0.5 * type_manual.item(), abs=1e-5)


def test_masked_rows_do_not_reach_the_type_head():
    bl, tl = _logits()
    tl = tl.clone().requires_grad_(True)
    y = torch.tensor([1, 0, 0, 0])
    targets = torch.zeros(4, N_TYPES)
    targets[0, 3] = 1.0
    mask = torch.tensor([True, False, False, False])
    arch.two_head_loss(bl, tl, y, targets, mask)[0].backward()
    assert tl.grad[0].abs().sum() > 0
    assert tl.grad[1:].abs().sum() == 0  # benign rows never train the type head


def test_no_attack_rows_means_binary_loss_only():
    bl, tl = _logits()
    y = torch.zeros(4)
    total, parts = arch.two_head_loss(
        bl, tl, y, torch.zeros(4, N_TYPES), torch.zeros(4, dtype=torch.bool)
    )
    assert parts["type"].item() == 0.0 and total.item() == pytest.approx(parts["binary"].item())


def test_class_weights_scale_the_type_loss():
    bl, tl = _logits()
    y = torch.ones(4)
    targets = torch.zeros(4, N_TYPES)
    targets[:, 1] = 1.0
    mask = torch.ones(4, dtype=torch.bool)
    base = arch.two_head_loss(bl, tl, y, targets, mask)[1]["type"]
    w = [1.0] * N_TYPES
    w[1] = 3.0
    weighted = arch.two_head_loss(bl, tl, y, targets, mask, type_class_weights=w)[1]["type"]
    assert weighted.item() == pytest.approx(3 * base.item(), rel=1e-5)


def test_both_losses_train_the_shared_encoder():
    model = _tiny_model().train()
    ids, mask = _batch()
    out = model(input_ids=ids, attention_mask=mask)
    y = torch.tensor([1, 0, 1, 1])
    targets = torch.zeros(4, N_TYPES)
    targets[0, 0] = targets[2, 1] = targets[3, 2] = 1.0
    type_mask = y.bool()

    def encoder_grad(loss_key):
        model.zero_grad()
        _, parts = arch.two_head_loss(out.binary_logit, out.type_logits, y, targets, type_mask)
        parts[loss_key].backward(retain_graph=True)
        return model.model.layers[0].attn.Wqkv.weight.grad.abs().sum().item()

    assert encoder_grad("binary") > 0 and encoder_grad("type") > 0


def test_loss_drives_a_toy_task():
    torch.manual_seed(0)
    model = _tiny_model().train()
    ids = torch.randint(3, 64, (16, 8))
    y = (ids[:, 0] % 2).long()  # label depends on the first token
    types = (ids[:, 0] % N_TYPES).long()
    targets = torch.nn.functional.one_hot(types, N_TYPES).float()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(60):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        loss, _ = arch.two_head_loss(out.binary_logit, out.type_logits, y, targets, y.bool())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < 0.5 * first


# ---- combination rule


def test_class_probabilities_sum_to_one_and_follow_the_formula():
    bl, tl = _logits()
    probs = arch.class_probabilities(bl, tl)
    assert probs.shape == (4, len(LABELS))
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
    p, q = torch.sigmoid(bl), torch.softmax(tl, -1)
    assert torch.allclose(probs[:, 0], 1 - p, atol=1e-6)
    assert torch.allclose(probs[:, 1:], p.unsqueeze(-1) * q, atol=1e-6)


def test_decide_gates_on_the_binary_head():
    tl = torch.zeros(3, N_TYPES)
    tl[:, 4] = 5.0  # type head strongly says type 4 for every row
    bl = torch.tensor([-3.0, 0.0, 3.0])  # clearly benign, exactly at the threshold, clearly attack
    ids, scores = arch.decide(bl, tl, threshold=0.5)
    assert ids.tolist() == [0, 5, 5]  # type 4 is LABELS[5]; the benign row ignores the type head
    assert LABELS[int(ids[1])] == arch.TYPE_LABELS[4]
    p, q_max = torch.sigmoid(bl), torch.softmax(tl, -1).max(-1).values
    assert scores[0].item() == pytest.approx(1 - p[0].item(), abs=1e-6)
    assert scores[2].item() == pytest.approx((p[2] * q_max[2]).item(), abs=1e-6)


def test_decide_differs_from_joint_argmax_by_design():
    # p_inj = 0.6 but the type is spread out: the joint argmax would say benign (0.4 > 0.6 * 0.15),
    # the rule says attack, because the binary head decides.
    bl = torch.tensor([math.log(0.6 / 0.4)])
    tl = torch.zeros(1, N_TYPES)
    ids, _ = arch.decide(bl, tl)
    assert ids.item() != 0
    assert arch.class_probabilities(bl, tl).argmax(-1).item() == 0


def test_threshold_is_respected():
    bl, tl = torch.tensor([1.0]), torch.zeros(1, N_TYPES)  # p_inj is about 0.73
    assert arch.decide(bl, tl, threshold=0.9)[0].item() == 0
    assert arch.decide(bl, tl, threshold=0.5)[0].item() != 0
