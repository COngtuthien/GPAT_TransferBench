"""GPAT (E08-E11) runtime/source contract infrastructure (M7C2a).

M7C2a holds only what qualifies the frozen runtime and source contract: the pinned NAFNet source loader
(`naf_source`), the R-04 teacher-preprocessing adapters (`teacher_preprocess`) and the reference forms of
the owner-frozen implementation clarifications (`runtime_contract`). The GPAT architecture, losses and
training loop are not implemented here (M7C2b and later).
"""
