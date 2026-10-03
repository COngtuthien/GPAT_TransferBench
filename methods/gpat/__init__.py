"""GPAT (E08-E11) implementation package.

Current state (M7C2b): the static GPAT core is implemented (config loader, wavelet, high-pass, E_art, heads/GRL,
NAFResidualUNet G_res, composition and artifact map, PatchGAN D, spectra, pure losses, EMA, schedule/batching/identity
helpers, GPATCore facade) on top of the M7C2a runtime/source infrastructure (`naf_source`, `teacher_preprocess`,
`runtime_contract`), and it is qualified on synthetic CPU tensors. GPU runtime qualification (incl. R-04 Level 2) is not
complete; no production training runner exists and scientific execution is not authorized.
"""
