# Verbatim excerpts of the AWQ reference implementation, used ONLY as a test oracle.
#
# Source: https://github.com/mit-han-lab/llm-awq @ d6e797a42b9ef7778de8ee2352116e0f48a78d61, MIT
#   (licence text: THIRD_PARTY_LICENSES/llm-awq-LICENSE.txt)
#   - awq/quantize/quantizer.py::pseudo_quantize_tensor
#   - awq/quantize/auto_clip.py::auto_clip_layer
#   - awq/quantize/auto_scale.py::get_act_scale, and _search_module_scale - a closure inside
#     auto_scale_block upstream, dedented to module level here. It reads `w_quantize_func`
#     from the enclosing scope; make_search() below rebuilds that scope exactly as
#     auto_scale_block does.
# Function bodies are byte-for-byte upstream. Do not "fix" them: their job is to be the reference.
# ruff: noqa
# fmt: off

import gc

import torch
import torch.nn as nn


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)



def make_search(w_bit, q_config):
    """auto_scale_block's preamble: bind w_quantize_func, return _search_module_scale."""
    def w_quantize_func(p):
        return pseudo_quantize_tensor(
            p,
            n_bit=w_bit,
            **q_config,
        ).detach()

    namespace = dict(globals(), w_quantize_func=w_quantize_func)
    exec(compile(_SEARCH_SOURCE, __file__, "exec"), namespace)
    return namespace["_search_module_scale"]


_SEARCH_SOURCE = r'''
# find the best scale ratio
def _search_module_scale(block, linears2scale: list, x, kwargs={}):
    # w: co, ci
    # x: n, ci
    x = x.to(next(block.parameters()).device)
    with torch.no_grad():
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    n_grid = 20
    history = []

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (
            (org_out - out).float().pow(2).mean().item()
        )  # float prevents overflow
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
        block.load_state_dict(org_sd)
    if best_ratio == -1:
        print(history)
        raise Exception
    # print(best_ratio)
    best_scales = best_scales.view(-1)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach()
'''
