"""First-order FP32 Metal GRU kernels for the experimental learner."""

import torch
from pathlib import Path

SRC = (
    Path(__file__).resolve().parents[2] / "native/learner/metal/gru.metal"
).read_text()
_lib = None


def library():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SRC)
    return _lib


class FusedGRU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, wi, wh, bi, bh, *, lengths=None):
        if (
            x.device.type != "mps"
            or x.dtype != torch.float32
            or x.ndim != 3
            or h.ndim != 2
            or h.shape[1] not in (32, 64)
            or x.shape[0] != h.shape[0]
            or x.shape[1] < 1
        ):
            raise ValueError(
                "Metal GRU requires FP32 MPS sequences and hidden size 32 or 64"
            )
        H = h.shape[1]
        I = x.shape[2]
        if (
            wi.shape != (3 * H, I)
            or wh.shape != (3 * H, H)
            or bi.shape != (3 * H,)
            or bh.shape != (3 * H,)
            or any(
                v.device.type != "mps" or v.dtype != torch.float32
                for v in (h, wi, wh, bi, bh)
            )
        ):
            raise ValueError("Metal GRU parameter shapes, device or dtype differ")
        x = x.contiguous()
        h = h.contiguous()
        wi = wi.contiguous()
        wh = wh.contiguous()
        bi = bi.contiguous()
        bh = bh.contiguous()
        B, T, I = x.shape
        H = h.shape[1]
        gi = torch.nn.functional.linear(x, wi, bi).contiguous()
        # Optional lengths are immutable counts from checked text/ordered recipes.
        # Tree calls forward/backward directly; the autograd apply API stays six-input.
        if lengths is not None:
            if (
                lengths.shape != (B,)
                or lengths.dtype != torch.int64
                or lengths.device != x.device
            ):
                raise ValueError(
                    "Metal GRU lengths must be one MPS int64 count per row"
                )
            lengths = lengths.contiguous()
        ctx.lengths = lengths
        allocate = torch.empty if lengths is None else torch.zeros
        states = allocate((B, T + 1, H), device=x.device)
        gates = allocate((B, T, 4 * H), device=x.device)
        if lengths is not None:
            kernel = (
                library().forward_gru_r16_lengths
                if B <= 2 and T > 256
                else library().forward_gru_simd_lengths
            )
            kernel(
                gi,
                wh,
                bh,
                h,
                states,
                gates,
                lengths,
                T,
                H,
                threads=B * H * 16,
                group_size=H * 16,
            )
        elif B <= 2 and T > 256:
            library().forward_gru_r16(
                gi,
                wh,
                bh,
                h,
                states,
                gates,
                T,
                H,
                threads=B * H * 16,
                group_size=H * 16,
            )
        else:
            library().forward_gru_simd(
                gi,
                wh,
                bh,
                h,
                states,
                gates,
                T,
                H,
                threads=B * H * 16,
                group_size=H * 16,
            )
        ctx.save_for_backward(x, states, gates, wi, wh)
        return states[:, 1:, :].contiguous()

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise ValueError("Metal GRU supports first-order gradients only")
        x, states, gates, wi, wh = ctx.saved_tensors
        B, T, I = x.shape
        H = wh.shape[1]
        lengths = ctx.lengths
        allocate = torch.empty if lengths is None else torch.zeros
        dgi = allocate((B, T, 3 * H), device=x.device)
        dgh = torch.empty_like(dgi) if lengths is None else torch.zeros_like(dgi)
        dh0 = torch.empty((B, H), device=x.device)
        kernel = (
            library().backward_gru
            if lengths is None
            else library().backward_gru_lengths
        )
        extra = () if lengths is None else (lengths,)
        kernel(
            grad.contiguous(),
            states,
            gates,
            wh,
            dgi,
            dgh,
            dh0,
            *extra,
            T,
            H,
            threads=B * H,
            group_size=H,
        )
        a = dgi.reshape(-1, 3 * H)
        b = dgh.reshape(-1, 3 * H)
        return (
            dgi @ wi,
            dh0,
            a.T @ x.reshape(-1, I),
            b.T @ states[:, :-1].reshape(-1, H),
            a.sum(0),
            b.sum(0),
        )


def gru(x, h, wi, wh, bi, bh):
    return FusedGRU.apply(x, h, wi, wh, bi, bh)
