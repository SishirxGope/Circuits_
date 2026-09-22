# Plan B checklist — print this and take it to the office

## Packed on the PC before travelling
- [ ] Gemma-2 and Llama-3.2 licences ACCEPTED and verified (takes days — do it first)
- [ ] `fetch_assets.py --plan b --datasets` completed
- [ ] `docs/baseline_pc.txt` and `docs/baseline_tests.txt` recorded
- [ ] `make_transfer_bundle.ps1` run; transfer folder copied
- [ ] Freeze-ownership table written into `docs/HUMAN_DECISIONS.md`

## First hour (install nothing yet)
- [ ] `00_inspect_hardware.sh` run; `docs/spark_environment.md` saved
- [ ] Architecture noted: aarch64 or x86_64
- [ ] Disk: 100 GB+ free

## Environment
- [ ] torch imports and `torch.cuda.is_available()` is True  ← fix before anything else
- [ ] Compute capability noted (needed for TORCH_CUDA_ARCH_LIST)
- [ ] `pip install -e .` succeeded
- [ ] GPTQ/AWQ: installed, OR fallback chosen and RECORDED in HUMAN_DECISIONS.md

## Restore
- [ ] Repo restored from bundle, history verified
- [ ] HF cache in place, `HF_HOME` exported
- [ ] `frozen/` copied (read-only — never regenerate)
- [ ] Upstream forks present

## The gate — no science until all four pass
- [ ] Test suite matches the PC baseline
- [ ] Pinned loader imports
- [ ] Pythia-160M attribution SCORES match the PC to tolerance (timings may differ)
- [ ] Frozen manifest verifies after transfer

## Timing and Q3
- [ ] All four models timed
- [ ] Q3 rule RE-APPLIED to the measured numbers
- [ ] Any change to B/S/R recorded BEFORE the freeze, never after

## GQA
- [ ] kv-heads check run on Gemma-2 and Llama-3.2
- [ ] If GQA: head-group mapping written and tested before Stage A

## Stage B — 88 cells
- [ ] Ran with `-P 1` first and verified one cell
- [ ] Scaled per-model, memory watched
- [ ] tmux/screen used
- [ ] All cells completed

## THE FREEZE — irreversible
- [ ] Ownership confirmed: THIS machine owns these cells
- [ ] Dry run read line by line
- [ ] Typed `FREEZE`
- [ ] Manifest hash recorded; `frozen/` committed locally

## Stage C, D, analysis
- [ ] All 88 Stage C cells ran, no hash mismatches
- [ ] CSI table with bootstrap CIs over B, S, r; BH at q = 0.05
- [ ] Both comparison levels reported
- [ ] Cross-audit against the FROZEN CSV (rho range -0.540..0.062; Gemma PPL 8.21)

## Home
- [ ] `08_bundle_results.sh` run; bundle verified
- [ ] `runs/` and `frozen/` carried back
- [ ] `frozen/` cells kept SEPARATE from the PC's — never merged
- [ ] No weights in the repo or the artifact
