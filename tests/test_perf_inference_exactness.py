"""Performance changes to fixed-weight inference must be bit-exact."""
import hashlib
import json
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.learner import contextual_model as m
from tests.test_contextual_dialogs import FAMILIES, panel, history_at


def reference_weights_sha256(model):
    """The original byte-list implementation, kept here as the exact oracle."""
    digest = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().clone()
        header = json.dumps((name, str(value.dtype), tuple(value.shape)), separators=(",", ":")).encode()
        data = bytes(value.reshape(-1).view(torch.uint8).tolist())
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def trace(model, frames, seed=5):
    """Every state, value, sampled action and conditional probability, exactly."""
    result = []
    encoder = m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema)
    state = model.initial_state()
    generator = torch.Generator().manual_seed(seed)
    for frame in frames:
        output = model.forward_step(encoder.encode(frame), state)
        state = output.state
        result.append(tuple(state.tolist()))
        if output.decision is not None:
            distribution = output.decision.distribution
            action = distribution.sample(generator=generator)
            evaluated = distribution.evaluate(action)
            result.append((output.decision.value.item(), action, evaluated.log_prob.item(),
                           tuple((x.name, x.selected_index, x.support_size, x.log_prob.item()) for x in evaluated.conditionals)))
    return result


class InferenceExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(301)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).eval().requires_grad_(False)
        self.histories = [history_at(panel(family)).frames for family in FAMILIES]

    def test_weights_digest_matches_original_byte_list_digest(self):
        self.assertEqual(m.weights_sha256(self.model), reference_weights_sha256(self.model))
        small = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.assertEqual(m.weights_sha256(small), reference_weights_sha256(small))
        with torch.no_grad():
            small.value.bias.add_(1)
        self.assertEqual(m.weights_sha256(small), reference_weights_sha256(small))

    def test_leaf_memo_reproduces_uncached_inference_exactly(self):
        # Autograd mode bypasses the memo; the computation is otherwise identical.
        with torch.enable_grad():
            expected = [trace(self.model, frames) for frames in self.histories]
        with torch.no_grad():
            first = [trace(self.model, frames) for frames in self.histories]
            second = [trace(self.model, frames) for frames in self.histories]
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertTrue(self.model.__dict__["_leaf_memo_state"][1]["text"])

    def test_memo_is_discarded_after_an_in_place_weight_change(self):
        with torch.no_grad():
            trace(self.model, self.histories[0])
            self.model.text_gru.weight_hh_l0.mul_(0.5)
            changed = trace(self.model, self.histories[0])
        fresh = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).eval().requires_grad_(False)
        fresh.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            self.assertEqual(changed, trace(fresh, self.histories[0]))

    def test_memo_never_replaces_differentiable_leaves(self):
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        frames = self.histories[0]
        with torch.no_grad():
            trace(model, frames)
        encoder = m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema)
        state = model.initial_state()
        for frame in frames:
            output = model.forward_step(encoder.encode(frame), state)
            state = output.state
        output.decision.value.backward()
        self.assertIsNotNone(model.byte.weight.grad)
        self.assertGreater(model.byte.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.text_gru.weight_ih_l0.grad)

    def test_signed_zero_leaves_stay_distinct(self):
        with torch.no_grad():
            memo = {"text": {}, "number": {}}
            positive = self.model._encode_tree(("number", 0.0), {m._LEAF_MEMO: memo})
            negative = self.model._encode_tree(("number", -0.0), {m._LEAF_MEMO: memo})
            self.assertEqual(len(memo["number"]), 2)
            self.assertTrue(torch.equal(positive, self.model._encode_tree(("number", 0.0), {})))
            self.assertTrue(torch.equal(negative, self.model._encode_tree(("number", -0.0), {})))


if __name__ == "__main__":
    unittest.main()
