# [AI-GEN] agent=Claude date=2026-09-30 task=B6 exit gate - the published IOI circuit as edges in our dense-node ids
# reviewed-by: PENDING
#
# Adapted from: https://github.com/ArthurConmy/Automatic-Circuit-Discovery @ bc99ace817974b5584b7ee203d596a8e2bbcd399, MIT
#   acdc/ioi/utils.py: IOI_CIRCUIT (copied verbatim, including the commented-out heads left
#   out) and get_ioi_true_edges (its three steps re-expressed over our edge ids), and
#   subnetwork_probing/train.py::iterative_correspondence_from_mask (what step 1 masks).
#   Licence: THIRD_PARTY_LICENSES/automatic-circuit-discovery-LICENSE.txt

"""The IOI circuit of Wang et al. (ICLR 2023) on GPT-2 small, as an edge set the gate can compare.

Wang et al. describe the circuit as 26 heads in 7 classes and the class-to-class
connections between them (Q, K or V). ACDC (Conmy et al., NeurIPS 2023) turned that
into the edge-level ground truth that ACDC and edge attribution patching are both
scored against (``get_ioi_true_edges``). This module is that function, over the same
edge universe our EAP scores (``src/extraction/eap.py::candidate_edges``):

1. **Mask every non-circuit head.** Start from every structurally possible edge and
   drop each edge into or out of the 118 heads outside ``IOI_CIRCUIT``
   (``iterative_correspondence_from_mask`` masks their q/k/v, their q/k/v inputs and,
   once all three are masked, their output). Every MLP edge and the embedding survive.
2. **Strip the circuit heads back to their MLP connections.** For each circuit head,
   remove embedding -> head and head -> head edges into its Q/K/V, and head -> logits.
3. **Add the published connections** (``special_connections``): INPUT is the embedding
   plus every MLP output, OUTPUT is the logits plus every MLP input, and an edge is added
   only when the receiving layer is later than the sending one (ACDC's layer numbers:
   embedding -1, logits 13).

ACDC's graph also has edges inside a component (``hook_q_input -> hook_q``,
``hook_q -> hook_result``, ``hook_mlp_in -> hook_mlp_out``). Our graph is component
outputs -> component inputs only (eap.py), so those are not part of this set.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from ..extraction.eap import EMB, LOGIT, candidate_edges
from ._acdc_vendored import ACDC_COMMIT

N_LAYERS = 12
N_HEADS = 12

# acdc/ioi/utils.py @ bc99ace8, verbatim (the heads upstream comments out stay out).
IOI_CIRCUIT: dict[str, list[tuple[int, int]]] = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

# get_ioi_true_edges' special_connections: (from class, to class, which inputs of the receiver).
SPECIAL_CONNECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("INPUT", "previous token", ("q", "k", "v")),
    ("INPUT", "duplicate token", ("q", "k", "v")),
    ("INPUT", "s2 inhibition", ("q",)),
    ("INPUT", "negative", ("k", "v")),
    ("INPUT", "name mover", ("k", "v")),
    ("INPUT", "backup name mover", ("k", "v")),
    ("previous token", "induction", ("k", "v")),
    ("induction", "s2 inhibition", ("k", "v")),
    ("duplicate token", "s2 inhibition", ("k", "v")),
    ("s2 inhibition", "negative", ("q",)),
    ("s2 inhibition", "name mover", ("q",)),
    ("s2 inhibition", "backup name mover", ("q",)),
    ("negative", "OUTPUT", ()),
    ("name mover", "OUTPUT", ()),
    ("backup name mover", "OUTPUT", ()),
)

SOURCE = (
    "Wang et al., ICLR 2023 (arXiv:2211.00593) IOI circuit, as the edge-level ground truth "
    f"of ACDC (Conmy et al., NeurIPS 2023): acdc/ioi/utils.py::get_ioi_true_edges @ {ACDC_COMMIT}, "
    "ported to dense-node edge ids by src/tasks/ioi_reference.py"
)

_HEAD = re.compile(r"^L(\d+)\.H(\d+)$")
_HEAD_INPUT = re.compile(r"^L(\d+)\.H(\d+)\.([QKV])$")


def _gpt2_small_edges() -> set[str]:
    cfg = SimpleNamespace(n_layers=N_LAYERS, n_heads=N_HEADS, parallel_attn_mlp=False)
    return set(candidate_edges(cfg))


def _head_of(node: str) -> tuple[int, int] | None:
    match = _HEAD.match(node) or _HEAD_INPUT.match(node)
    return (int(match.group(1)), int(match.group(2))) if match else None


def ioi_reference_edges() -> set[str]:
    """``get_ioi_true_edges`` in our ids (component edges only; see the module docstring)."""
    circuit = {head for heads in IOI_CIRCUIT.values() for head in heads}
    if len(circuit) != 26:  # upstream asserts the same
        raise AssertionError(f"IOI_CIRCUIT has {len(circuit)} distinct heads, expected 26")
    universe = _gpt2_small_edges()

    # 1. mask every head outside the circuit
    present = set()
    for edge in universe:
        src, dst = edge.split("->")
        if any(h is not None and h not in circuit for h in (_head_of(src), _head_of(dst))):
            continue
        present.add(edge)

    # 2. circuit heads keep only their MLP inputs, and lose their direct path to the logits
    for edge in list(present):
        src, dst = edge.split("->")
        into_head = bool(_HEAD_INPUT.match(dst)) and (src == EMB or bool(_HEAD.match(src)))
        head_to_logits = dst == LOGIT and bool(_HEAD.match(src))
        if into_head or head_to_logits:
            present.discard(edge)

    # 3. the published class-to-class connections
    def senders(group: str) -> list[tuple[int, str]]:
        if group == "INPUT":
            return [(-1, EMB)] + [(layer, f"L{layer}.MLP") for layer in range(N_LAYERS)]
        return [(layer, f"L{layer}.H{head}") for layer, head in IOI_CIRCUIT[group]]

    def receivers(group: str, qkv: tuple[str, ...]) -> list[tuple[int, str]]:
        if group == "OUTPUT":
            return [(13, LOGIT)] + [(layer, f"L{layer}.MLP") for layer in range(N_LAYERS)]
        return [
            (layer, f"L{layer}.H{head}.{letter.upper()}")
            for layer, head in IOI_CIRCUIT[group]
            for letter in qkv
        ]

    for inp, out, qkv in SPECIAL_CONNECTIONS:
        for layer_from, src in senders(inp):
            for layer_to, dst in receivers(out, qkv):
                if layer_to > layer_from:
                    edge = f"{src}->{dst}"
                    if edge not in universe:  # upstream would raise KeyError here
                        raise AssertionError(f"{edge} is not an edge of GPT-2 small")
                    present.add(edge)
    return present


__all__ = ["IOI_CIRCUIT", "SOURCE", "SPECIAL_CONNECTIONS", "ioi_reference_edges"]
