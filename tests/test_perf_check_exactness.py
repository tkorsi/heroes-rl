"""The plain state traversal and typed bit words must decide exactly as state_dict and byte comparison."""
from collections import OrderedDict
import unittest

import torch
from torch import nn

from experiments.learner import contextual_model as m
from tests.test_perf_inference_exactness import Tensors, reference_weights_sha256


def model(seed=7):
    torch.manual_seed(seed)
    return m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))


def described(state):
    """Every name with its tensor's storage, offset, dtype, shape, strides and device, in order."""
    return [(name, value.untyped_storage().data_ptr(), value.storage_offset(), value.dtype, tuple(value.shape),
             value.stride(), value.device) for name, value in state.items()]


class Extra(nn.Linear):
    def get_extra_state(self):
        return {"extra": 1}

    def set_extra_state(self, state):
        pass


class OwnSave(nn.Linear):
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "renamed"] = self.weight.detach()


class OwnStateDict(nn.Linear):
    def state_dict(self, *args, **kwargs):
        return OrderedDict()


class Odd(nn.Parameter):
    pass


class PlainStateTests(unittest.TestCase):
    def test_plain_state_equals_state_dict_names_order_and_aliases(self):
        net = model()
        net.register_buffer("kept", torch.arange(3.0))
        net.register_buffer("dropped", torch.arange(2.0), persistent=False)
        net.byte.register_parameter("absent", None)
        nested = nn.Sequential(nn.Linear(2, 3), nn.Sequential(nn.Tanh(), nn.Linear(3, 1)))
        nested.add_module("gap", None)
        net.add_module("nested", nested)
        plain = m._plain_state(net)
        self.assertIsNotNone(plain)
        self.assertEqual(described(plain), described(net.state_dict()))
        self.assertNotIn("dropped", plain)
        self.assertNotIn("byte.absent", plain)

    def test_anything_beyond_plain_module_code_takes_state_dict(self):
        cases = []
        hooked = model(); hooked.register_state_dict_post_hook(lambda module, state, prefix, local: None); cases.append(hooked)
        pre = model(); pre.byte.register_state_dict_pre_hook(lambda module, prefix, keep: None); cases.append(pre)
        for cls in (Extra, OwnSave, OwnStateDict):
            net = model(); net.value = cls(net.value.in_features, 1); cases.append(net)
        instance = model(); instance.byte._save_to_state_dict = lambda *args: None; cases.append(instance)
        own = model(); own.state_dict = lambda *args, **kwargs: OrderedDict(); cases.append(own)
        odd = model(); odd.value.weight = Odd(odd.value.weight.detach().clone()); cases.append(odd)
        buffered = model(); buffered.register_buffer("view", torch.arange(4.0).as_subclass(Odd)); cases.append(buffered)
        for index, net in enumerate(cases):
            with self.subTest(case=index):
                self.assertIsNone(m._plain_state(net))

    def test_snapshots_decide_as_the_digest_on_every_fallback(self):
        for scale in (2.0, 1.0):
            net = model()
            snapshot = m.WeightsSnapshot(net)
            # A post hook that rewrites the state: plain traversal must not be used.
            net.register_state_dict_post_hook(lambda module, state, prefix, local, scale=scale: state.__setitem__(
                "byte.weight", state["byte.weight"] * scale))
            expected = reference_weights_sha256(net) == snapshot.digest
            self.assertEqual(snapshot.unchanged(net), expected)
            self.assertEqual(expected, scale == 1.0)


class TypedWordsTests(unittest.TestCase):
    def test_typed_words_equal_exactly_when_bits_equal(self):
        base = torch.tensor([0.0, 1.0, float("nan"), float("inf"), 1e-40])
        other = base.clone()
        words = m._typed_words([base, base.view(5, 1)])
        self.assertEqual(words.dtype, torch.int64)  # 40 bytes: whole int64 words
        self.assertEqual(m._typed_words([base]).dtype, torch.int32)  # 20 bytes: words of the element width
        self.assertTrue(torch.equal(words, m._typed_words([other, other.view(5, 1)])))
        for index, value in ((0, -0.0), (2, None)):
            changed = base.clone()
            if value is None:
                changed.view(torch.int32)[index] ^= 1  # another NaN payload
            else:
                changed[index] = value
            self.assertTrue(changed[index] == base[index] or (changed[index] != changed[index]))
            self.assertFalse(torch.equal(m._typed_words([changed, base.view(5, 1)]), words))

    def test_typed_words_are_copies_and_refuse_mixed_or_strided_inputs(self):
        value = torch.ones(4)
        words = m._typed_words([value])
        value.add_(1)
        self.assertTrue(torch.equal(words, torch.ones(4).view(torch.int64)))  # the snapshot never aliases
        self.assertIsNone(m._typed_words([]))
        self.assertIsNone(m._typed_words([torch.ones(2), torch.ones(2, dtype=torch.float64)]))
        self.assertIsNone(m._typed_words([torch.ones(2, 3).t()]))
        self.assertIsNone(m._typed_words([torch.ones(2, dtype=torch.complex64)]))
        self.assertEqual(m._typed_words([torch.tensor([True, False])]).dtype, torch.uint8)

    def test_a_single_view_starting_inside_a_word_decides_as_the_digest(self):
        base = torch.arange(10.0)
        net = Tensors(x=base[1:5])  # contiguous, four floats from storage offset 1: not on an int64 boundary
        self.assertEqual(net.x.storage_offset() * net.x.element_size() % 8, 4)
        snapshot = m.WeightsSnapshot(net)
        self.assertIsNotNone(snapshot.typed)
        self.assertTrue(snapshot.matches(net))
        self.assertTrue(snapshot.unchanged(net))
        words = m._joined_words([net.x], False)
        self.assertEqual(words.dtype, torch.int64)
        base.view(torch.int32)[2] ^= 1  # one bit of the second buffer element
        self.assertFalse(snapshot.matches(net))
        self.assertEqual(snapshot.unchanged(net), reference_weights_sha256(net) == snapshot.digest)
        self.assertFalse(snapshot.unchanged(net))

    def test_bit_states_name_their_representation(self):
        uniform = [torch.ones(2), torch.zeros(3)]
        strided = [torch.ones(2), torch.zeros(3, 2).t()]
        self.assertEqual(m._bit_state(uniform)[0][-1], "typed")
        self.assertEqual(m._bit_state(strided)[0][-1], "bytes")
        self.assertEqual(m._bit_state(strided)[1].dtype, torch.int64)
        # Same values and meta in either layout: meta differs only by the marker, never equal words by accident.
        contiguous = [torch.ones(2), torch.zeros(2, 3)]
        self.assertNotEqual(m._bit_state(contiguous)[0], m._bit_state(strided)[0])

    def test_snapshot_typed_and_byte_paths_agree_on_mixed_models(self):
        floats = torch.tensor([[0.0, -0.0, float("nan")], [1.0, -3.5, 65504.0]])
        mixed = Tensors(f32=floats, f16=floats.half(), flag=torch.tensor([True, False]))
        uniform = Tensors(a=floats, b=floats.t().contiguous())
        for net in (mixed, uniform):
            snapshot = m.WeightsSnapshot(net)
            self.assertEqual(snapshot.typed is None, net is mixed)
            self.assertTrue(snapshot.matches(net))
            first = next(iter(net.buffers()))
            first.view(torch.int16 if first.element_size() == 2 else torch.int32).view(-1)[2] ^= 1
            self.assertFalse(snapshot.matches(net))
            self.assertEqual(snapshot.unchanged(net), reference_weights_sha256(net) == snapshot.digest)


if __name__ == "__main__":
    unittest.main()
