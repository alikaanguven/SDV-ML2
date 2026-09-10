"""Demonstrate pair-bias and padding-mask semantics in attention.

This mirrors the convention used in networks/ParT_K.py:

    input mask:       True = real track, False = padded track
    key padding mask: 0 for real keys, -9 for padded keys

The padding mask is a key mask. It suppresses attention to padded tracks by
adding -9 to the corresponding key columns before the softmax.
"""

import math

import torch
import torch.nn as nn


def manual_attention(q, k, v, pair_bias, real_track_mask):
    """Compute softmax(QK^T/sqrt(d_k) + B_pair + M_pad) V."""
    d_k = q.size(-1)

    padding_mask = (~real_track_mask).to(q.dtype).masked_fill(~real_track_mask, float("-9"))
    padding_mask = padding_mask[:, None, :]  # broadcast over query positions

    logits = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    logits = logits + pair_bias + padding_mask
    weights = torch.softmax(logits, dim=-1)
    output = weights @ v
    return output, weights, logits, padding_mask


def pytorch_attention_with_identity_projections(x, pair_bias, real_track_mask):
    """Use nn.MultiheadAttention with Q=K=V=x and identity projections."""
    batch_size, seq_len, embed_dim = x.shape

    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=1,
        dropout=0.0,
        bias=False,
        batch_first=True,
        dtype=x.dtype,
    )

    with torch.no_grad():
        eye = torch.eye(embed_dim, dtype=x.dtype)
        mha.in_proj_weight.copy_(torch.cat([eye, eye, eye], dim=0))
        mha.out_proj.weight.copy_(eye)

    key_padding_mask = (~real_track_mask).to(x.dtype).masked_fill(~real_track_mask, float("-9"))

    output, weights = mha(
        x,
        x,
        x,
        attn_mask=pair_bias.expand(batch_size, seq_len, seq_len),
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    return output, weights.squeeze(1), key_padding_mask


def main():
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(7)

    batch_size = 1
    seq_len = 4
    embed_dim = 3

    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float64)

    # Track 0, 1, and 2 are real. Track 3 is padding.
    real_track_mask = torch.tensor([[True, True, True, False]])

    # Zero padded input, as in the model before the attention block.
    x = x.masked_fill(~real_track_mask[..., None], 0.0)

    # One head, one batch. B_pair has shape (batch, query_track, key_track).
    # This matrix is symmetric, as in the default ParT pair-embedding path.
    pair_bias_2d = torch.tensor(
        [
            [0.00, 0.20, -0.10, 5.00],
            [0.20, 0.00, 0.10, 5.00],
            [-0.10, 0.10, 0.00, 5.00],
            [5.00, 5.00, 5.00, 5.00],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(pair_bias_2d, pair_bias_2d.T)
    pair_bias = pair_bias_2d.unsqueeze(0)

    manual_out, manual_weights, logits, m_pad = manual_attention(
        q=x,
        k=x,
        v=x,
        pair_bias=pair_bias,
        real_track_mask=real_track_mask,
    )
    torch_out, torch_weights, key_padding_mask = pytorch_attention_with_identity_projections(
        x=x,
        pair_bias=pair_bias,
        real_track_mask=real_track_mask,
    )

    print("Input real-track mask, where True means real and False means padded:")
    print(real_track_mask)
    print()

    print("M_pad added to attention logits. The padded key column is -9:")
    print(m_pad[0])
    print()

    print("Pair bias B_pair. The padded key has a large positive bias before masking:")
    print(pair_bias[0])
    print()

    print("Final logits = QK^T/sqrt(d_k) + B_pair + M_pad:")
    print(logits[0])
    print()

    print("Attention weights after softmax:")
    print(manual_weights[0])
    print()

    print("Manual attention output:")
    print(manual_out[0])
    print()

    print("PyTorch MultiheadAttention output:")
    print(torch_out[0])
    print()

    print("PyTorch key_padding_mask passed to nn.MultiheadAttention:")
    print(key_padding_mask)
    print()

    torch.testing.assert_close(manual_weights, torch_weights)
    torch.testing.assert_close(manual_out, torch_out)

    print("Checks passed:")
    print("  manual weights == PyTorch weights")
    print("  manual output  == PyTorch output")


if __name__ == "__main__":
    main()
