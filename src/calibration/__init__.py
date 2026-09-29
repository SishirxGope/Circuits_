# [AI-GEN] agent=Claude date=2026-09-29 task=Q7 calibration package
# reviewed-by: PENDING

"""Calibration data for the compressors that need it (Q7, configs/calibration/final.yaml).

``token_cache``   - the fixed token array every calibrated compressor sees.
``second_moment`` - per-projection input statistics E[x^2] collected over that array
                    (Wanda's score; the same forward pass GPTQ/AWQ calibration needs).
"""
