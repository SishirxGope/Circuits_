# [AI-GEN] agent=Claude date=2026-09-30 task=B6 - ACDC's IOI ground truth, run verbatim as the oracle for src/tasks/ioi_reference.py
# reviewed-by: PENDING
#
# GENERATED FILE - do not edit by hand. Verbatim excerpts, extracted by a script (ast, not
# transcribed), of https://github.com/ArthurConmy/Automatic-Circuit-Discovery
# @ bc99ace817974b5584b7ee203d596a8e2bbcd399, MIT
# (licence: THIRD_PARTY_LICENSES/automatic-circuit-discovery-LICENSE.txt):
#   acdc/TLACDCEdge.py              EdgeType, Edge, TorchIndex
#   acdc/TLACDCInterpNode.py        TLACDCInterpNode
#   acdc/acdc_utils.py              OrderedDefaultdict, make_nd_dict
#   acdc/TLACDCCorrespondence.py    TLACDCCorrespondence
#   subnetwork_probing/train.py     iterative_correspondence_from_mask
#   acdc/ioi/utils.py               IOI_CIRCUIT, Conn, get_ioi_true_edges
# The one edit is marked [edit]: an in-function import of a function defined in this file.
# ruff: noqa
"""ACDC's own construction of the IOI ground-truth edges, runnable without the ACDC package."""

from __future__ import annotations  # upstream annotations name HookedTransformer; never evaluated

import collections
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

class EdgeType(Enum):
    """
    Property of edges in the computational graph - either 
    
    ADDITION: the child (hook_name, index) is a sum of the parent (hook_name, index)s
    DIRECT_COMPUTATION The *single* child is a function of and only of the parent (e.g the value hooked by hook_q is a function of what hook_q_input saves).
    PLACEHOLDER generally like 2. but where there are generally multiple parents. Here in ACDC we just include these edges by default when we find them. Explained below?
    
    Q: Why do we do this?

    There are two answers to this question: A1 is an interactive notebook, see <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/main/notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb">this Colab notebook</a>, which is in this repo at notebooks/implementation_demo.py. A2 is an answer that is written here below, but probably not as clear as A1 (though shorter).

    A2: We need something inside TransformerLens to represent the edges of a computational graph.
    The object we choose is pairs (hook_name, index). For example the output of Layer 11 Heads is a hook (blocks.11.attn.hook_result) and to sepcify the 3rd head we add the index [:, :, 3]. Then we can build a computational graph on these! 

    However, when we do ACDC there turn out to be two conflicting things "removing edges" wants to do: 
    i) for things in the residual stream, we want to remove the sum of the effects from previous hooks 
    ii) for things that are not linear we want to *recompute* e.g the result inside the hook 
    blocks.11.attn.hook_result from a corrupted Q and normal K and V

    The easiest way I thought of of reconciling these different cases, while also having a connected computational graph, is to have three types of edges: addition for the residual case, direct computation for easy cases where we can just replace hook_q with a cached value when we e.g cut it off from hook_q_input, and placeholder to make the graph connected (when hook_result is connected to hook_q and hook_k and hook_v)"""

    ADDITION = 0
    DIRECT_COMPUTATION = 1
    PLACEHOLDER = 2
    
    def __eq__(self, other):
        """Necessary because of extremely frustrating error that arises with load_ext autoreload (because this uses importlib under the hood: https://stackoverflow.com/questions/66458864/enum-comparison-become-false-after-reloading-module)"""

        assert isinstance(other, EdgeType)
        return self.value == other.value


class Edge:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
        effect_size: Optional[float] = None,
    ):
        self.edge_type = edge_type
        self.present = present
        self.effect_size = effect_size

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"


class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `TorchIndex`s are essentially indices that say which part of the tensor is being affected. 

    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])    

    Also we want to be able to call e.g `my_dictionary[my_torch_index]` hence the hashable tuple stuff
    
    Note: ideally this would be integrated with transformer_lens.utils.Slice in future; they are accomplishing similar but different things"""

    def __init__(
        self, 
        list_of_things_in_tuple: List,
    ):
        # check correct types
        for arg in list_of_things_in_tuple:
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        # make an object that can be indexed into a tensor
        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])

        # make an object that can be hashed (so used as a dictionary key)
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    # some graphics things

    def __repr__(self, use_actual_colon=True) -> str: # graphviz, an old library used to dislike actual colons in strings, but this shouldn't be an issue anymore
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":" if use_actual_colon else "COLON"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

    def graphviz_index(self, use_actual_colon=True) -> str:
        return self.__repr__(use_actual_colon=use_actual_colon)


class TLACDCInterpNode:
    """Represents one node in the computational graph, similar to ACDCInterpNode from the rust_circuit code
    
    But WARNING this has nodes closer to the input tokens as *parents* of nodes closer to the output tokens, the opposite of the rust_circuit code
    
    Params:
        name: name of the node
        index: the index of the tensor that this node represents
        mode: how we deal with this node when we bump into it as a parent of another node. Addition: it's summed to make up the child. Direct_computation: it's the sole node used to compute the child. Off: it's not the parent of a child ever."""
        
    def __init__(self, name: str, index: TorchIndex, incoming_edge_type: EdgeType):
        
        self.name = name
        self.index = index
        
        self.parents: List["TLACDCInterpNode"] = []
        self.children: List["TLACDCInterpNode"] = []

        self.incoming_edge_type = incoming_edge_type

    def _add_child(self, child_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.children.append(child_node)

    def _add_parent(self, parent_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.parents.append(parent_node)

    def __repr__(self):
        return f"TLACDCInterpNode({self.name}, {self.index})"

    def __str__(self) -> str:
        index_str = "" if len(self.index.hashable_tuple) < 3 else f"_{self.index.hashable_tuple[2]}"
        return f"{self.name}{self.index}"


class OrderedDefaultdict(defaultdict):
    def __init__(self, *args, **kwargs):
        if sys.version_info < (3, 7):
            raise Exception("You need Python >= 3.7 so dict is ordered by default. You could revert to the old unmantained implementation https://github.com/ArthurConmy/Automatic-Circuit-Discovery/commit/65301ec57c31534bd34383c243c782e3ccb7ed82")
        super().__init__(*args, **kwargs)


def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))


class TLACDCCorrespondence:
    """Stores the full computational graph, similar to ACDCCorrespondence from the rust_circuit code
    
    The two attributes, self.graph and self.edges allow for efficiently looking up the nodes and edges in the graph: see `notebooks/editing_edges.py`"""

    graph: MutableMapping[str, MutableMapping[TorchIndex, TLACDCInterpNode]]
    edges: MutableMapping[str, MutableMapping[TorchIndex, MutableMapping[str, MutableMapping[TorchIndex, Edge]]]]
    def __init__(self):
        self.graph = OrderedDefaultdict(OrderedDict) # TODO rename "nodes?"
        self.edges = make_nd_dict(end_type=None, n=4)

    def first_node(self):
        return self.graph[list(self.graph.keys())[0]][list(self.graph[list(self.graph.keys())[0]].keys())[0]]

    def nodes(self) -> List[TLACDCInterpNode]:
        """Concatenate all nodes in the graph"""
        return [node for by_index_list in self.graph.values() for node in by_index_list.values()]
    
    def all_edges(self) -> Dict[Tuple[str, TorchIndex, str, TorchIndex], Edge]:
        """Concatenate all edges in the graph"""
        
        big_dict = {}

        for child_name, rest1 in self.edges.items():
            for child_index, rest2 in rest1.items():
                for parent_name, rest3 in rest2.items():
                    for parent_index, edge in rest3.items():
                        assert edge is not None, (child_name, child_index, parent_name, parent_index, "Edges have been setup WRONG somehow...")

                        big_dict[(child_name, child_index, parent_name, parent_index)] = edge
        
        return big_dict

    def add_node(self, node: TLACDCInterpNode, safe=True):
        if safe:
            assert node not in self.nodes(), f"Node {node} already in graph"
        self.graph[node.name][node.index] = node

    def add_edge(
        self,
        parent_node: TLACDCInterpNode,
        child_node: TLACDCInterpNode,
        edge: Edge,
        safe=True,
    ):
        if safe:
            if parent_node not in self.nodes(): # TODO could be slow ???
                self.add_node(parent_node)
            if child_node not in self.nodes():
                self.add_node(child_node)
        
        assert child_node.incoming_edge_type == edge.edge_type, (child_node.incoming_edge_type, edge.edge_type)
        
        parent_node._add_child(child_node)
        child_node._add_parent(parent_node)

        self.edges[child_node.name][child_node.index][parent_node.name][parent_node.index] = edge
    
    def remove_edge(
        self,
        child_name: str,
        child_index: TorchIndex,
        parent_name: str,
        parent_index: TorchIndex,
    ):
        try:
            edge = self.edges[child_name][child_index][parent_name][parent_index]
        except Exception as e:
            print("Couldn't index in - are you sure this edge exists???")
            raise e

        edge.present=False
        del self.edges[child_name][child_index][parent_name][parent_index]

        # more efficiency things...
        if len(self.edges[child_name][child_index][parent_name]) == 0:
            del self.edges[child_name][child_index][parent_name]
        if len(self.edges[child_name][child_index]) == 0:
            del self.edges[child_name][child_index]
        if len(self.edges[child_name]) == 0:
            del self.edges[child_name]

        parent = self.graph[parent_name][parent_index]
        child = self.graph[child_name][child_index]

        parent.children.remove(child)
        child.parents.remove(parent)        

    @classmethod
    def setup_from_model(cls, model, use_pos_embed=False):
        correspondence = cls()

        downstream_residual_nodes: List[TLACDCInterpNode] = []
        logits_node = TLACDCInterpNode(
            name=f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
            index=TorchIndex([None]),
            incoming_edge_type = EdgeType.ADDITION,
        )
        correspondence.add_node(logits_node) 
        downstream_residual_nodes.append(logits_node)

        for layer_idx in range(model.cfg.n_layers - 1, -1, -1):
            # connect MLPs
            if not model.cfg.attn_only: 
                # this MLP writed to all future residual stream things
                cur_mlp_name = f"blocks.{layer_idx}.hook_mlp_out"
                cur_mlp_slice = TorchIndex([None])
                cur_mlp = TLACDCInterpNode(
                    name=cur_mlp_name,
                    index=cur_mlp_slice,
                    incoming_edge_type=EdgeType.PLACEHOLDER,
                )
                correspondence.add_node(cur_mlp)
                for residual_stream_node in downstream_residual_nodes:
                    correspondence.add_edge(
                        parent_node=cur_mlp,
                        child_node=residual_stream_node,
                        edge=Edge(edge_type=EdgeType.ADDITION),
                        safe=False,
                    )

                cur_mlp_input_name = f"blocks.{layer_idx}.hook_mlp_in"
                cur_mlp_input_slice = TorchIndex([None])
                cur_mlp_input = TLACDCInterpNode(
                    name=cur_mlp_input_name,
                    index=cur_mlp_input_slice,
                    incoming_edge_type=EdgeType.ADDITION,
                )
                correspondence.add_node(cur_mlp_input)
                correspondence.add_edge(
                    parent_node=cur_mlp_input,
                    child_node=cur_mlp,
                    edge=Edge(edge_type=EdgeType.PLACEHOLDER), # EDIT: previously, this was a DIRECT_COMPUTATION edge, but that leads to overcounting of MLP edges (I think)
                    safe=False,
                )

                downstream_residual_nodes.append(cur_mlp_input)

            new_downstream_residual_nodes: List[TLACDCInterpNode] = []

            # connect attention heads
            for head_idx in range(model.cfg.n_heads - 1, -1, -1):
                # this head writes to all future residual stream things
                cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
                cur_head_slice = TorchIndex([None, None, head_idx])
                cur_head = TLACDCInterpNode(
                    name=cur_head_name,
                    index=cur_head_slice,
                    incoming_edge_type=EdgeType.PLACEHOLDER,
                )
                correspondence.add_node(cur_head)
                for residual_stream_node in downstream_residual_nodes:
                    correspondence.add_edge(
                        parent_node=cur_head,
                        child_node=residual_stream_node,
                        edge=Edge(edge_type=EdgeType.ADDITION),
                        safe=False,
                    )

                for letter in "qkv":
                    hook_letter_name = f"blocks.{layer_idx}.attn.hook_{letter}"
                    hook_letter_slice = TorchIndex([None, None, head_idx])
                    hook_letter_node = TLACDCInterpNode(name=hook_letter_name, index=hook_letter_slice, incoming_edge_type=EdgeType.DIRECT_COMPUTATION)
                    correspondence.add_node(hook_letter_node)

                    hook_letter_input_name = f"blocks.{layer_idx}.hook_{letter}_input"
                    hook_letter_input_slice = TorchIndex([None, None, head_idx])
                    hook_letter_input_node = TLACDCInterpNode(
                        name=hook_letter_input_name, index=hook_letter_input_slice, incoming_edge_type=EdgeType.ADDITION
                    )
                    correspondence.add_node(hook_letter_input_node)

                    correspondence.add_edge(
                        parent_node = hook_letter_node,
                        child_node = cur_head,
                        edge = Edge(edge_type=EdgeType.PLACEHOLDER),
                        safe = False,
                    )

                    correspondence.add_edge(
                        parent_node=hook_letter_input_node,
                        child_node=hook_letter_node,
                        edge=Edge(edge_type=EdgeType.DIRECT_COMPUTATION),
                        safe=False,
                    )

                    new_downstream_residual_nodes.append(hook_letter_input_node)
            downstream_residual_nodes.extend(new_downstream_residual_nodes)

        if use_pos_embed:
            token_embed_node = TLACDCInterpNode(
                name="hook_embed",
                index=TorchIndex([None]),
                incoming_edge_type=EdgeType.PLACEHOLDER,
            )
            pos_embed_node = TLACDCInterpNode(
                name="hook_pos_embed",
                index=TorchIndex([None]),
                incoming_edge_type=EdgeType.PLACEHOLDER,
            )
            embed_nodes = [token_embed_node, pos_embed_node]

        else:
            # add the embedding node
            embedding_node = TLACDCInterpNode(
                name="blocks.0.hook_resid_pre",
                index=TorchIndex([None]),
                incoming_edge_type=EdgeType.PLACEHOLDER, # TODO maybe add some NoneType or something???
            )
            embed_nodes = [embedding_node]

        for embed_node in embed_nodes:
            correspondence.add_node(embed_node)
            for node in downstream_residual_nodes:
                correspondence.add_edge(
                    parent_node=embed_node,
                    child_node=node,
                    edge=Edge(edge_type=EdgeType.ADDITION),
                    safe=False,
                )
    
        return correspondence

    def count_no_edges(self, verbose=False):
        cnt = 0

        for tupl, edge in self.all_edges().items():
            if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                cnt += 1
                if verbose:
                    print(tupl)

        if verbose:
            print("No edge", cnt)
        return cnt


def iterative_correspondence_from_mask(
    model: HookedTransformer,
    nodes_to_mask: List[TLACDCInterpNode], # Can be empty
    use_pos_embed: bool = False,
    corr: Optional[TLACDCCorrespondence] = None,
    head_parents: Optional[List] = None,
) -> Tuple[TLACDCCorrespondence, List]:
    """Given corr has some nodes masked, also mask the nodes_to_mask"""

    assert (corr is None) == (head_parents is None), "Ensure we're either masking from scratch or we provide details on `head_parents`"

    if corr is None:
        corr = TLACDCCorrespondence.setup_from_model(model, use_pos_embed=use_pos_embed)
    if head_parents is None:
        head_parents = collections.defaultdict(lambda: 0)

    additional_nodes_to_mask = []

    for node in nodes_to_mask:
        additional_nodes_to_mask.append(
            TLACDCInterpNode(node.name.replace(".attn.", ".") + "_input", node.index, EdgeType.ADDITION)
        )

        if node.name.endswith("_q") or node.name.endswith("_k") or node.name.endswith("_v"):
            child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
            head_parents[(child_name, node.index)] += 1

            if head_parents[(child_name, node.index)] == 3:
                additional_nodes_to_mask.append(TLACDCInterpNode(child_name, node.index, EdgeType.PLACEHOLDER))

            # Forgot to add these in earlier versions of Subnetwork Probing, and so the edge counts were inflated
            additional_nodes_to_mask.append(TLACDCInterpNode(child_name + "_input", node.index, EdgeType.ADDITION))

        if node.name.endswith(("mlp_in", "resid_mid")):
            additional_nodes_to_mask.append(
                TLACDCInterpNode(
                    node.name.replace("resid_mid", "mlp_out").replace("mlp_in", "mlp_out"),
                    node.index,
                    EdgeType.DIRECT_COMPUTATION,
                )
            )

    assert all([v <= 3 for v in head_parents.values()]), "We should have at most three parents (Q, K and V, connected via placeholders)"

    for node in nodes_to_mask + additional_nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                if node.name in rest2 and node.index in rest2[node.name]:
                    rest2[node.name][node.index].present = False

    return corr, head_parents


IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}


@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]


def get_ioi_true_edges(model):
    nodes_to_mask = []
    
    all_groups_of_nodes = [group for _, group in IOI_CIRCUIT.items()]
    all_nodes = [node for group in all_groups_of_nodes for node in group]
    assert len(all_nodes) == 26, len(all_nodes)

    nodes_to_mask = []

    for layer_idx in range(12):
        for head_idx in range(12):
            if (layer_idx, head_idx) not in all_nodes:
                for letter in ["q", "k", "v"]:
                    nodes_to_mask.append(
                        TLACDCInterpNode(name=f"blocks.{layer_idx}.attn.hook_{letter}", index = TorchIndex([None, None, head_idx]), incoming_edge_type=EdgeType.DIRECT_COMPUTATION),
                    )


    # [edit] was: from subnetwork_probing.train import iterative_correspondence_from_mask (defined above)
    corr, _ = iterative_correspondence_from_mask(
        nodes_to_mask = nodes_to_mask,
        model = model,
    )

    # For all heads...
    for layer_idx, head_idx in all_nodes:
        for letter in "qkv":
            # remove input -> head connection
            edge_to = corr.edges[f"blocks.{layer_idx}.hook_{letter}_input"][TorchIndex([None, None, head_idx])]
            edge_to[f"blocks.0.hook_resid_pre"][TorchIndex([None])].present = False

            # Remove all other_head->this_head connections in the circuit
            for layer_from in range(layer_idx):
                for head_from in range(12):
                    edge_to[f"blocks.{layer_from}.attn.hook_result"][TorchIndex([None, None, head_from])].present = False

            # Remove connection from this head to the output
            corr.edges["blocks.11.hook_resid_post"][TorchIndex([None])][f"blocks.{layer_idx}.attn.hook_result"][TorchIndex([None, None, head_idx])].present = False



    special_connections: set[Conn] = {
        Conn("INPUT", "previous token", ("q", "k", "v")),
        Conn("INPUT", "duplicate token", ("q", "k", "v")),
        Conn("INPUT", "s2 inhibition", ("q",)),
        Conn("INPUT", "negative", ("k", "v")),
        Conn("INPUT", "name mover", ("k", "v")),
        Conn("INPUT", "backup name mover", ("k", "v")),
        Conn("previous token", "induction", ("k", "v")),
        Conn("induction", "s2 inhibition", ("k", "v")),
        Conn("duplicate token", "s2 inhibition", ("k", "v")),
        Conn("s2 inhibition", "negative", ("q",)),
        Conn("s2 inhibition", "name mover", ("q",)),
        Conn("s2 inhibition", "backup name mover", ("q",)),
        Conn("negative", "OUTPUT", ()),
        Conn("name mover", "OUTPUT", ()),
        Conn("backup name mover", "OUTPUT", ()),
    }

    for conn in special_connections:
        if conn.inp == "INPUT":
            idx_from = [(-1, "blocks.0.hook_resid_pre", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_from.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_out", TorchIndex([None])))
        else:
            idx_from = [(layer_idx, f"blocks.{layer_idx}.attn.hook_result", TorchIndex([None, None, head_idx])) for layer_idx, head_idx in IOI_CIRCUIT[conn.inp]]

        if conn.out == "OUTPUT":
            idx_to = [(13, "blocks.11.hook_resid_post", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_to.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_in", TorchIndex([None])))
        else:
            idx_to = [
                (layer_idx, f"blocks.{layer_idx}.hook_{letter}_input", TorchIndex([None, None, head_idx]))
                for layer_idx, head_idx in IOI_CIRCUIT[conn.out]
                for letter in conn.qkv
            ]

        for layer_from, layer_name_from, which_idx_from in idx_from:
            for layer_to, layer_name_to, which_idx_to in idx_to:
                if layer_to > layer_from:
                    corr.edges[layer_name_to][which_idx_to][layer_name_from][which_idx_from].present = True

    ret =  OrderedDict({(t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present for t, e in corr.all_edges().items() if e.present})
    return ret
