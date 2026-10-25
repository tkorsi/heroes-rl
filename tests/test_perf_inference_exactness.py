"""Performance changes to fixed-weight inference must be bit-exact."""
from copy import deepcopy
import hashlib
import json
import pickle
import unittest

try:
    import torch
    from torch import nn
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
    return trace_sessions(model, [frames], [seed])[0]


def trace_sessions(model, histories, seeds):
    """Interleave separate players frame by frame; each keeps its own codec, state and RNG."""
    sessions = [(m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema), torch.Generator().manual_seed(seed))
                for frames, seed in zip(histories, seeds)]
    states, results = [model.initial_state() for _ in histories], [[] for _ in histories]
    for index in range(max(map(len, histories))):
        for player, frames in enumerate(histories):
            if index >= len(frames):
                continue
            (encoder, generator), result = sessions[player], results[player]
            output = model.forward_step(encoder.encode(frames[index]), states[player])
            states[player] = output.state
            result.append(tuple(output.state.tolist()))
            if output.decision is not None:
                distribution = output.decision.distribution
                action = distribution.sample(generator=generator)
                evaluated = distribution.evaluate(action)
                result.append((output.decision.value.item(), action, evaluated.log_prob.item(),
                               tuple((x.name, x.selected_index, x.support_size, x.log_prob.item()) for x in evaluated.conditionals)))
    return results


def unmemoized(model):
    """A fresh model with identical weights, dtype and mode that never consults a memo."""
    fresh = m.ContextualActorCritic(model.config).to(model.byte.weight.dtype).train(model.training).requires_grad_(False)
    fresh.load_state_dict(model.state_dict())
    fresh._leaf_memo = lambda: None
    return fresh


def entries(model):
    return model.__dict__["_leaf_memo_state"][3]


class Tensors(nn.Module):
    def __init__(self, **values):
        super().__init__()
        self.config = m.ModelConfig()
        for name, value in values.items():
            self.register_buffer(name, value)


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

    def assertMatchesFresh(self, model, frames=None):
        frames = self.histories[0] if frames is None else frames
        with torch.no_grad():
            result = trace(model, frames)
            self.assertEqual(result, trace(unmemoized(model), frames))
        return result

    def test_weights_digest_matches_original_byte_list_digest(self):
        self.assertEqual(m.weights_sha256(self.model), reference_weights_sha256(self.model))
        small = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        self.assertEqual(m.weights_sha256(small), reference_weights_sha256(small))
        with torch.no_grad():
            small.value.bias.add_(1)
        self.assertEqual(m.weights_sha256(small), reference_weights_sha256(small))
        self.assertEqual(m.weights_sha256(self.model.double()), reference_weights_sha256(self.model))

    def test_weights_digest_matches_original_across_dtypes_and_shapes(self):
        floats = torch.tensor([[0.0, -0.0, float("nan"), float("inf")], [-float("inf"), 1e-40, -3.5, 65504.0]])
        values = dict(f32=floats, f64=floats.double(), f16=floats.half(), bf16=floats.bfloat16(),
                      i64=torch.tensor([-2 ** 63, -1, 0, 2 ** 63 - 1]), flag=torch.tensor([True, False, True]),
                      u8=torch.arange(256, dtype=torch.uint8).reshape(16, 16), empty=torch.zeros(0, 3),
                      empty_i64=torch.zeros(0, dtype=torch.long), scalar=torch.tensor(-0.0), scalar_i64=torch.tensor(7),
                      scalar_flag=torch.tensor(False), transposed=torch.arange(6.).reshape(2, 3).t(), sliced=torch.arange(10)[::3])
        module = Tensors(**values)
        self.assertEqual(m.weights_sha256(module), reference_weights_sha256(module))
        for name, value in values.items():
            single = Tensors(**{name: value})
            self.assertEqual(m.weights_sha256(single), reference_weights_sha256(single), name)
        self.assertEqual(m.weights_sha256(Tensors()), reference_weights_sha256(Tensors()))
        self.assertNotEqual(m.weights_sha256(Tensors(x=torch.tensor(0.0))), m.weights_sha256(Tensors(x=torch.tensor(-0.0))))

    def test_leaf_memo_reproduces_uncached_inference_exactly(self):
        # Autograd mode bypasses the memo; the computation is otherwise identical.
        with torch.enable_grad():
            expected = [trace(self.model, frames) for frames in self.histories]
        with torch.no_grad():
            first = [trace(self.model, frames) for frames in self.histories]
            second = [trace(self.model, frames) for frames in self.histories]
            self.assertEqual([trace(unmemoized(self.model), frames) for frames in self.histories], expected)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertTrue(entries(self.model)["text"])
        with torch.inference_mode():
            self.assertEqual([trace(self.model, frames) for frames in self.histories], expected)
            self.assertEqual([trace(self.model, frames) for frames in self.histories], expected)

    def test_paired_histories_keep_separate_player_state(self):
        histories, seeds = self.histories[:2], (11, 12)
        with torch.no_grad():
            alone = [trace(unmemoized(self.model), frames, seed) for frames, seed in zip(histories, seeds)]
            self.assertNotEqual(alone[0], alone[1])
            self.assertEqual(trace_sessions(self.model, histories, seeds), alone)  # fresh memo
            self.assertEqual(trace_sessions(self.model, histories, seeds), alone)  # prefilled memo
            self.assertEqual(trace_sessions(self.model, histories[::-1], seeds[::-1]), alone[::-1])
            self.assertEqual([trace(self.model, frames, seed) for frames, seed in zip(histories, seeds)], alone)

    def test_memo_is_discarded_after_an_in_place_weight_change(self):
        with torch.no_grad():
            trace(self.model, self.histories[0])
            self.model.text_gru.weight_hh_l0.mul_(0.5)
            changed = trace(self.model, self.histories[0])
        fresh = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).eval().requires_grad_(False)
        fresh.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            self.assertEqual(changed, trace(fresh, self.histories[0]))

    def assertInvalidatedBy(self, change):
        before = self.assertMatchesFresh(self.model)
        stale = entries(self.model)
        with torch.no_grad():
            change()
        after = self.assertMatchesFresh(self.model)
        self.assertIsNot(entries(self.model), stale)
        self.assertNotEqual(after, before)

    def test_memo_is_discarded_after_data_mutation_without_version_change(self):
        weight = self.model.byte.weight
        version = weight._version
        self.assertInvalidatedBy(lambda: weight.data.mul_(-1.5))
        self.assertEqual(weight._version, version)
        self.assertInvalidatedBy(lambda: self.model.number[0].bias.data.add_(0.25))

    def test_memo_is_discarded_after_data_replacement_even_at_the_same_address(self):
        weight = self.model.text_gru.weight_hh_l0
        self.assertInvalidatedBy(lambda: setattr(weight, "data", torch.randn_like(weight)))
        address, version, parameter = weight.data_ptr(), weight._version, id(weight)

        def alias():
            same = torch.empty(0).set_(weight.untyped_storage(), weight.storage_offset(), weight.shape, weight.stride())
            weight.data = same
            same.copy_(torch.randn_like(same))
        self.assertInvalidatedBy(alias)
        self.assertEqual((weight.data_ptr(), weight._version, id(weight)), (address, version, parameter))

    def test_memo_is_discarded_after_signed_zero_or_nan_payload_bit_changes(self):
        bias = self.model.number[0].bias
        with torch.no_grad():
            bias[0] = 0.0
            trace(self.model, self.histories[0])
            stale = entries(self.model)
            bias.data[0] = -0.0
            self.assertIsNot(self.model._leaf_memo(), stale)
            stale = entries(self.model)
            bias.data.view(torch.int32)[1] = 0x7FC00001
            self.assertIsNot(self.model._leaf_memo(), stale)
            stale = entries(self.model)
            bias.data.view(torch.int32)[1] = 0x7FC00002
            self.assertIsNot(self.model._leaf_memo(), stale)
            self.assertIs(self.model._leaf_memo(), entries(self.model))

    def test_memo_is_discarded_after_copy_vector_and_optimizer_updates(self):
        weight = self.model.text_gru.weight_ih_l0
        self.assertInvalidatedBy(lambda: weight.data.copy_(torch.randn_like(weight)))
        leaves = [self.model.byte.weight, *self.model.text_gru.parameters(), *self.model.number.parameters()]
        vector = nn.utils.parameters_to_vector(leaves)
        self.assertInvalidatedBy(lambda: nn.utils.vector_to_parameters(vector.flip(0), leaves))
        for options in (dict(foreach=True), dict(fused=True), dict(foreach=False)):
            self.model.requires_grad_(True)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05, **options)

            def step():
                with torch.enable_grad():
                    for parameter in self.model.parameters():
                        parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
            self.assertInvalidatedBy(step)
            self.model.requires_grad_(False)

    def test_memo_is_bypassed_under_autocast(self):
        expected = self.assertMatchesFresh(self.model)
        stale = entries(self.model)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertIsNone(self.model._leaf_memo())
            self.assertEqual(trace(self.model, self.histories[0]), trace(unmemoized(self.model), self.histories[0]))
            self.assertNotEqual(trace(self.model, self.histories[0]), expected)
        self.assertIs(entries(self.model), stale)
        self.assertEqual(self.assertMatchesFresh(self.model), expected)

    def test_memo_follows_dtype_changes(self):
        original = self.assertMatchesFresh(self.model)
        self.model.to(torch.float64)
        widened = self.assertMatchesFresh(self.model)
        self.assertNotEqual(widened, original)
        self.assertEqual(next(iter(entries(self.model)["text"].values())).dtype, torch.float64)
        self.model.to(torch.float32)
        self.assertEqual(self.assertMatchesFresh(self.model), original)
        self.model.half()
        self.assertMatchesFresh(self.model)
        self.assertEqual(next(iter(entries(self.model)["text"].values())).dtype, torch.float16)

    def test_memo_is_discarded_after_checkpoint_loads(self):
        torch.manual_seed(302)
        other = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).state_dict()
        self.assertInvalidatedBy(lambda: self.model.load_state_dict(other))
        torch.manual_seed(303)
        other = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA)).requires_grad_(False).state_dict()
        weight = self.model.byte.weight
        self.assertInvalidatedBy(lambda: self.model.load_state_dict(other, assign=True))
        self.assertIsNot(self.model.byte.weight, weight)

    def test_memo_is_discarded_after_parameter_or_module_replacement(self):
        gru, number = self.model.text_gru, self.model.number
        self.assertInvalidatedBy(lambda: setattr(number[0], "weight", nn.Parameter(torch.randn_like(number[0].weight), requires_grad=False)))
        self.assertInvalidatedBy(lambda: setattr(gru, "bias_ih_l0", nn.Parameter(torch.randn_like(gru.bias_ih_l0), requires_grad=False)))
        self.assertIs(gru._flat_weights[gru._flat_weights_names.index("bias_ih_l0")], gru.bias_ih_l0)
        replacement = nn.Embedding(257, 8, padding_idx=0).requires_grad_(False).eval()
        self.assertInvalidatedBy(lambda: setattr(self.model, "byte", replacement))

    def test_memo_is_not_used_in_training_mode_and_survives_no_change_across_modes(self):
        self.assertMatchesFresh(self.model)
        stale = entries(self.model)
        self.model.train()
        with torch.no_grad():
            self.assertIsNone(self.model._leaf_memo())
            self.model.byte.weight.data.mul_(2)  # like an optimizer step while training
        self.assertMatchesFresh(self.model.train())
        self.assertIs(entries(self.model), stale)  # untouched in training mode
        self.model.eval()
        self.assertMatchesFresh(self.model)
        self.assertIsNot(entries(self.model), stale)
        current = entries(self.model)
        self.model.train(); self.model.eval()
        self.assertMatchesFresh(self.model)
        self.assertIs(entries(self.model), current)  # identical bits: reuse is exact
        self.model.text_gru.train()
        with torch.no_grad():
            self.assertIsNone(self.model._leaf_memo())
        self.model.text_gru.eval()
        handle = self.model.number.register_forward_hook(lambda module, inputs, output: output * 2)
        with torch.no_grad():
            self.assertIsNone(self.model._leaf_memo())
        handle.remove()

    def test_copies_and_pickles_never_carry_a_memo(self):
        before = self.assertMatchesFresh(self.model)
        for copied in (deepcopy(self.model), pickle.loads(pickle.dumps(self.model))):
            self.assertNotIn("_leaf_memo_state", copied.__dict__)
            self.assertEqual(self.assertMatchesFresh(copied), before)
            self.assertIsNot(entries(copied), entries(self.model))
            copied.text_gru.weight_ih_l0.data.mul_(-1)
            self.assertNotEqual(self.assertMatchesFresh(copied), before)
            self.assertEqual(self.assertMatchesFresh(self.model), before)
        copied = deepcopy(self.model)
        self.model.byte.weight.data.mul_(-1)
        self.assertNotEqual(self.assertMatchesFresh(self.model), before)
        self.assertEqual(self.assertMatchesFresh(copied), before)

    def test_memo_is_revalidated_before_sampling_a_distribution(self):
        def decide(model, mutate):
            frames = self.histories[0]
            encoder = m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema)
            state, decision = model.initial_state(), None
            for frame in frames:
                output = model.forward_step(encoder.encode(frame), state)
                state, decision = output.state, output.decision or decision
            mutate(model)
            distribution = decision.distribution
            action = distribution.sample(generator=torch.Generator().manual_seed(9))
            evaluated = distribution.evaluate(action)
            return action, evaluated.log_prob.item(), [(x.name, x.log_prob.item()) for x in evaluated.conditionals]

        def mutate(model):
            model.byte.weight.data.mul_(-2)
            model.number[0].weight.data.mul_(-2)
        with torch.no_grad():
            trace(self.model, self.histories[0])
            reference = unmemoized(self.model)
            stale = entries(self.model)
            self.assertEqual(decide(self.model, mutate), decide(reference, mutate))
            self.assertIsNot(entries(self.model), stale)

    def test_inference_mode_entries_stay_in_inference_mode(self):
        with torch.inference_mode():
            inferred = trace(self.model, self.histories[0])
            self.assertTrue(next(iter(entries(self.model)["text"].values())).is_inference())
        with torch.no_grad():
            self.assertEqual(trace(self.model, self.histories[0]), inferred)
            self.assertFalse(any(value.is_inference() for kind in entries(self.model).values() for value in kind.values()))

    def test_memo_never_replaces_differentiable_leaves(self):
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4)).eval()
        frames = self.histories[0]
        with torch.no_grad():
            trace(model, frames)
        with torch.inference_mode():
            trace(model, frames)

        class Trap(dict):
            def get(self, *args):
                raise AssertionError("memo consulted")
        model.__dict__["_leaf_memo_state"] = model.__dict__["_leaf_memo_state"][:3] + ({"text": Trap(), "number": Trap()},)
        with torch.inference_mode(), self.assertRaisesRegex(AssertionError, "memo consulted"):
            trace(model, frames)  # the trap is live wherever the memo is consulted
        encoder = m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema)
        state = model.initial_state()
        for frame in frames:
            output = model.forward_step(encoder.encode(frame), state)
            state = output.state
        distribution = output.decision.distribution
        (output.decision.value + distribution.evaluate(distribution.sample(generator=torch.Generator().manual_seed(1))).log_prob).backward()
        for parameter in (model.byte.weight, model.text_gru.weight_ih_l0, model.number[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(model._leaf_memo())

    def test_signed_zero_leaves_stay_distinct(self):
        with torch.no_grad():
            memo = {"text": {}, "number": {}}
            positive = self.model._encode_tree(("number", 0.0), {m._LEAF_MEMO: memo})
            negative = self.model._encode_tree(("number", -0.0), {m._LEAF_MEMO: memo})
            self.assertEqual(len(memo["number"]), 2)
            self.assertTrue(torch.equal(positive, self.model._encode_tree(("number", 0.0), {})))
            self.assertTrue(torch.equal(negative, self.model._encode_tree(("number", -0.0), {})))

    def test_leaf_keys_keep_text_number_and_type_distinctions(self):
        with torch.no_grad():
            memo = {"text": {}, "number": {}}
            for value in (1, 1.0, True, 0, False, 0.0):
                self.assertTrue(torch.equal(self.model._encode_tree(("number", value), {m._LEAF_MEMO: memo}),
                                            self.model._encode_tree(("number", value), {})))
            self.assertEqual(len(memo["number"]), 6)
            self.model._encode_tree(("text", "1"), {m._LEAF_MEMO: memo})
            self.assertEqual(list(memo["text"]), ["1"])
            nan = self.model._encode_tree(("number", float("nan")), {m._LEAF_MEMO: memo})
            self.assertEqual(len(memo["number"]), 6)  # NaN payloads are never shared
            self.assertTrue(torch.equal(m._bits(nan), m._bits(self.model._encode_tree(("number", float("nan")), {}))))


    def test_internal_node_memo_reproduces_uncached_inference_exactly(self):
        with torch.no_grad():
            expected = [trace(unmemoized(self.model), frames) for frames in self.histories]
            self.assertEqual([trace(self.model, frames) for frames in self.histories], expected)
            nodes = entries(self.model)["node"]
            self.assertTrue(nodes)
            filled = len(nodes)
            self.assertEqual([trace(self.model, frames) for frames in self.histories], expected)  # served from the memo
            self.assertEqual(len(nodes), filled)
        with torch.inference_mode():
            self.assertEqual([trace(self.model, frames) for frames in self.histories], expected)

    def test_internal_node_memo_is_discarded_after_tree_module_changes(self):
        self.assertInvalidatedBy(lambda: self.model.tree[0].weight.data.mul_(-1.5))
        self.assertInvalidatedBy(lambda: self.model.ordered.weight_hh_l0.data.mul_(-1.5))

    def test_bounded_internal_node_memo_stays_exact(self):
        limit = m._NODE_MEMO_LIMIT
        try:
            m._NODE_MEMO_LIMIT = 3
            self.assertMatchesFresh(self.model)
            self.assertLessEqual(len(entries(self.model)["node"]), 3)
            self.assertMatchesFresh(self.model)
        finally:
            m._NODE_MEMO_LIMIT = limit

    def roots(self, trees, *, memo):
        """Encode top-level trees with one frame cache; the reference uses no memo at all."""
        with torch.no_grad():
            cache = {m._LEAF_MEMO: self.model._leaf_memo()} if memo else {}
            encode = self.model._encode_root if memo else self.model._encode_tree
            values = [m._bits(encode(tree, cache)).tolist() for tree in trees]
            keys = sorted(repr(key) for key in cache if type(key) is tuple)
            return values, keys, [m._bits(cache[key]).tolist() for key in sorted((k for k in cache if type(k) is tuple), key=repr)]

    def test_inexact_roots_meet_the_reference_frame_cache(self):
        number = lambda value: m._node("count", m._number(1), ("number", value))
        exact = m._node("root", number(0.0), m._node("inner", number(2.0), m._text("a")))
        signed = m._node("other", number(-0.0), m._node("inner", number(2.0), m._text("a")))
        nan = m._node("nan", ("number", float("nan")), number(0.0))
        integer = m._node("int", ("number", 0), number(0.0))
        self.assertTrue(m._exact_tree(exact))
        size = self.model.config.feature_size

        class SignedLeaf(nn.Module):
            """Keeps the input's signed zero (Linear accumulators do not), so a non-reference cache shows."""
            def forward(self, value):
                return value[:1].repeat(size)
        self.model.number = SignedLeaf().eval()
        with torch.no_grad():
            self.assertFalse(torch.equal(m._bits(self.model._encode_tree(("number", 0.0), {})),
                                         m._bits(self.model._encode_tree(("number", -0.0), {}))))
        for tree in (signed, nan, integer, ("number", 1), ("text", 1), ("node", "x"), ("node", 1, ())):
            self.assertFalse(m._exact_tree(tree), tree)
        with torch.no_grad():
            self.model._encode_root(exact, {m._LEAF_MEMO: self.model._leaf_memo()})  # an earlier frame fills the memo
        self.assertTrue(entries(self.model)["node"])
        for trees in ((exact, signed), (signed, exact), (exact, nan, signed), (integer, exact), (exact, integer, exact)):
            with self.subTest(trees=[tree[1] for tree in trees]):
                self.assertEqual(self.roots(trees, memo=True), self.roots(trees, memo=False))

        class SignedOrdered(nn.Module):
            def forward(self, sequence):  # the last element's encoding as the final hidden state
                return sequence, sequence[:, -1:].transpose(0, 1)

        class SignedTree(nn.Module):
            def forward(self, joined):  # the pooled half, so a child's signed zero reaches the node value
                return joined[size:]
        # Linear and GRU accumulators drop signed zeros; these modules keep them, so an internal node whose memo
        # entry came from an equal tree with the other zero would show in the root value itself.
        self.model.ordered, self.model.tree = SignedOrdered().eval(), SignedTree().eval()
        sequence = lambda value: ("ordered", "sequence", (("number", value),))
        positive, negative = ("ordered", "first", (sequence(0.0),)), ("ordered", "second", (sequence(-0.0),))
        with torch.no_grad():
            self.assertFalse(torch.equal(m._bits(self.model._encode_tree(positive, {})),
                                         m._bits(self.model._encode_tree(negative, {}))))
            self.model._encode_root(positive, {m._LEAF_MEMO: self.model._leaf_memo()})
        for trees in ((negative,), (negative, positive), (positive, negative)):
            with self.subTest(trees=[tree[1] for tree in trees]):
                self.assertEqual(self.roots(trees, memo=True), self.roots(trees, memo=False))

    def test_a_revalidated_memo_never_seeds_entries_from_another_weight_version(self):
        frames = self.histories[0]
        with torch.no_grad():
            trace(self.model, frames)
            encoder = m.ContextualEncoder(frames[0].episode, frames[0].player, frames[0].schema)
            state = self.model.initial_state()
            for frame in frames:
                output = self.model.forward_step(encoder.encode(frame), state)
                state, decision = output.state, output.decision or locals().get("decision")
            distribution = decision.distribution
            self.model.tree[0].weight.data.mul_(-1)
            distribution.sample(generator=torch.Generator().manual_seed(3))
            self.assertIs(distribution.cache[m._EXACT_CACHE], False)
            self.model.tree[0].weight.data.mul_(-1)
            self.assertMatchesFresh(self.model)


if __name__ == "__main__":
    unittest.main()
