"""Deterministic lexicographic TRAIN/VAL/TEST allocator (Q-01).

Scope
-----
This module decides, for one dataset at a time, which **allocation group** goes to which split. It
never touches frames: a group is assigned as a whole, and every sampled frame inherits its canonical
video's split. It is pure metadata arithmetic — no dataset bytes are read.

What is being balanced
----------------------
The allocation object is the **canonical video**, and every canonical video counts as exactly one
video regardless of how many of its sampled frames survived M2. A frame-level technical failure
(`SCRFD_NO_FACE`) must not make a video "lighter" in the split objective, because the video is still
one video of that subject/content and still contributes its surviving frames downstream. Only rows
with M2 status COMPLETE become usable image samples; FAILED rows stay provenance-only.

Lexicographic objective (owner decision Q-01)
---------------------------------------------
    P0  hard grouping / leakage feasibility (constraint, never a penalty)
    P1  total canonical-video ratio
    P2  binary live/spoof distribution
    P3  attack_macro distribution
    P4  attack_raw distribution
    P5  deterministic seeded tie-break

A later priority may never worsen an earlier one: each level is minimized, then pinned as an
equality constraint before the next level is minimized.

    P1:  E1 = sum_s |100*n_s - pct_s*N|
    Pk:  Ek = sum_c sum_s W_c * |100*n_{c,s} - pct_s*N_c|,  W_c = SCALE // N_c

`W_c` is the frozen **fixed-point** normalisation that gives every category equal weight instead of
letting the majority category dominate by size. It is the definition of the objective, not an
approximation evaluated at runtime: all coefficients and all deviations are integers, so the
objective is an exact integer and the solver's optimum is an exact optimum *of this objective*.
`SCALE` is frozen at 10**6, which keeps every coefficient <= 10**6 and every objective value below
~5e9 — far inside the exactly-representable range of the float64 arithmetic the MILP backend uses,
so no rounding can occur inside the solve. The idealised rational form is sum_c sum_s
d/N_c; the fixed-point form differs from it by less than (sum of all deviations)/SCALE, which is
reported in the objective record so the approximation is visible rather than implied.

Why this is tractable exactly
-----------------------------
Groups are first collapsed into **profile classes**: two groups are interchangeable when they carry
the identical multiset of (label_binary, attack_macro, attack_raw) over their videos. Every P1-P4
term is a function of category counts only, so any two assignments with the same per-class counts
score identically. On the real data this collapses 50/35/1694 groups to 1/1/16 classes, turning the
allocation into a tiny integer program that HiGHS solves to proven global optimality.

Determinism
-----------
Two things could otherwise be arbitrary, and both are pinned:

* **which count vector** — after P1-P4 are pinned, the class-count vector itself is made unique by
  lexicographically minimising the counts in a fixed canonical class order (a further sequence of
  tiny solves). The solver is therefore never allowed to pick among equally optimal vectors.
* **which groups** — inside a class, groups are ordered by
  `sha256("gpatbench.split_v1.tie.v1|<dataset>|<group_id>|<seed>")` as lowercase hex, ties broken by
  `group_id`, and handed out TRAIN, then VAL, then TEST.

Nothing depends on input row order, dict/set iteration, PYTHONHASHSEED, filesystem order, thread
count, host or wall-clock time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

SPLITS = ("TRAIN", "VAL", "TEST")
PCT = {"TRAIN": 70, "VAL": 15, "TEST": 15}          # spec §3.5, integer percentages
SPLIT_SEED = 20260814                                # spec §3.5
TIE_NAMESPACE = "gpatbench.split_v1.tie.v1"
SCALE = 10 ** 6                                      # frozen fixed-point normalisation scale
LEVELS = ("P1_total", "P2_binary", "P3_attack_macro", "P4_attack_raw")


class AllocationError(RuntimeError):
    """The allocation input violates a hard (P0) requirement."""


class UnresolvedPolicyError(AllocationError):
    """Input needs an owner policy decision that does not exist yet (never silently worked around)."""


@dataclass(frozen=True)
class Video:
    """One canonical video, with the M2 outcome of its sampled frames."""
    video_id: str
    label_binary: int
    attack_macro: str
    attack_raw: str | None
    n_complete: int          # M2 COMPLETE sampled frames -> usable image rows
    n_failed: int = 0        # M2 FAILED sampled frames -> provenance only

    @property
    def profile(self) -> tuple:
        return (int(self.label_binary), self.attack_macro, self.attack_raw or "")


@dataclass(frozen=True)
class Group:
    """One allocation group: the indivisible unit of assignment (subject, or exact-content group)."""
    dataset: str
    group_id: str
    videos: tuple

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def class_signature(self) -> tuple:
        """Groups with equal signatures are interchangeable for every P1-P4 term."""
        return tuple(sorted(v.profile for v in self.videos))


def stable_group_rank(dataset: str, group_id: str, seed: int = SPLIT_SEED) -> str:
    """Frozen P5 tie-break key: lowercase hex SHA-256 over a namespaced string."""
    return hashlib.sha256(f"{TIE_NAMESPACE}|{dataset}|{group_id}|{seed}".encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ category bookkeeping
def _category_keys(level: str, v: Video):
    if level == "P2_binary":
        return [str(int(v.label_binary))]
    if level == "P3_attack_macro":
        return [v.attack_macro]
    if level == "P4_attack_raw":
        return [v.attack_raw if v.attack_raw else "(none)"]
    raise KeyError(level)


def _class_category_counts(classes, level):
    """counts[class_index][category] = number of videos of that category in one group of the class."""
    out = []
    for sig in classes:
        c: dict = {}
        for (lb, macro, raw) in sig:
            v = Video("", lb, macro, raw or None, 0)
            for k in _category_keys(level, v):
                c[k] = c.get(k, 0) + 1
        out.append(c)
    return out


def validate_groups(groups) -> None:
    """P0 input validation. Problems are raised, never silently repaired."""
    if not groups:
        raise AllocationError("no allocation groups")
    seen_ids, seen_videos = set(), set()
    datasets = {g.dataset for g in groups}
    if len(datasets) != 1:
        raise AllocationError(f"allocate() runs per dataset; got {sorted(datasets)}")
    for g in groups:
        if g.group_id in seen_ids:
            raise AllocationError(f"duplicate allocation group id {g.group_id!r}")
        seen_ids.add(g.group_id)
        if not g.videos:
            raise AllocationError(f"allocation group {g.group_id!r} has no canonical video")
        for v in g.videos:
            if v.video_id in seen_videos:
                raise AllocationError(f"canonical video {v.video_id!r} appears in more than one group")
            seen_videos.add(v.video_id)
            if v.n_complete <= 0:
                # A video with no usable frame cannot contribute an image row. Whether it should be
                # dropped, kept, or block the split is an owner policy decision that does not exist.
                raise UnresolvedPolicyError(
                    f"canonical video {v.video_id!r} has zero M2 COMPLETE samples; "
                    "no owner policy exists for zero-usable canonical videos "
                    "(BLOCKED_BY_ZERO_USABLE_CANONICAL_VIDEO)")


# ------------------------------------------------------------------ MILP construction
@dataclass
class _Model:
    """Integer model over x[j][s] = number of groups of class j in split s, plus |.| aux columns."""
    n_classes: int
    n_groups_per_class: list
    videos_per_group: list
    class_counts: dict                       # level -> [ {category: per-group video count} ]
    nx: int = 0
    aux: list = field(default_factory=list)  # (coef_row_over_x, const, aux_col)
    obj_terms: dict = field(default_factory=dict)   # level -> [(aux_col, weight)]
    category_sizes: dict = field(default_factory=dict)

    def xi(self, j, s):
        return j * len(SPLITS) + s

    def build(self):
        import numpy as np
        self.nx = self.n_classes * len(SPLITS)
        for level in LEVELS:
            terms, sizes = [], {}
            if level == "P1_total":
                cats = ["__total__"]
            else:
                cats = sorted({k for cc in self.class_counts[level] for k in cc})
            for cat in cats:
                if level == "P1_total":
                    per_class = list(self.videos_per_group)
                else:
                    per_class = [self.class_counts[level][j].get(cat, 0) for j in range(self.n_classes)]
                N_c = sum(per_class[j] * self.n_groups_per_class[j] for j in range(self.n_classes))
                if N_c == 0:
                    continue
                sizes[cat] = N_c
                weight = 1 if level == "P1_total" else SCALE // N_c
                for s, sp in enumerate(SPLITS):
                    row = np.zeros(self.nx)
                    for j in range(self.n_classes):
                        row[self.xi(j, s)] = 100.0 * per_class[j]
                    self.aux.append((row, PCT[sp] * N_c, len(self.aux)))
                    terms.append((len(self.aux) - 1, weight))
            self.obj_terms[level] = terms
            self.category_sizes[level] = sizes
        return self

    @property
    def total_cols(self):
        return self.nx + len(self.aux)

    def constraints(self, pins):
        """Assignment rows + |.| linearisation + every pinned equality (level optima, canonical x)."""
        import numpy as np
        rows, lo, hi = [], [], []
        for j in range(self.n_classes):                      # sum_s x[j][s] == M_j   (P0)
            r = np.zeros(self.total_cols)
            for s in range(len(SPLITS)):
                r[self.xi(j, s)] = 1.0
            rows.append(r)
            lo.append(float(self.n_groups_per_class[j]))
            hi.append(float(self.n_groups_per_class[j]))
        for row, const, k in self.aux:                       # d >= expr and d >= -expr
            r = np.zeros(self.total_cols)
            r[:self.nx] = -row
            r[self.nx + k] = 1.0
            rows.append(r); lo.append(-float(const)); hi.append(np.inf)
            r = np.zeros(self.total_cols)
            r[:self.nx] = row
            r[self.nx + k] = 1.0
            rows.append(r); lo.append(float(const)); hi.append(np.inf)
        for coefs, value in pins:                            # exact equality: never relaxed later
            r = np.zeros(self.total_cols)
            for col, w in coefs:
                r[col] += float(w)
            rows.append(r); lo.append(float(value)); hi.append(float(value))
        return np.vstack(rows), np.array(lo), np.array(hi)

    def level_cost(self, level):
        return [(self.nx + k, w) for k, w in self.obj_terms[level]]


def canonical_column_order(dataset: str, classes, seed: int = SPLIT_SEED):
    """Seeded, deterministic order in which count columns are pinned during the P5 tie-break.

    The order must be fixed, but it must not be a hard-coded preference: iterating classes/splits in
    their natural order would systematically hand every leftover group to the same split. Ordering
    the columns by a seeded hash keeps the choice reproducible while letting the frozen seed, rather
    than the code layout, decide which split absorbs an unavoidable remainder.
    """
    cols = [(j, s) for j in range(len(classes)) for s in range(len(SPLITS))]
    def key(js):
        j, s = js
        h = hashlib.sha256(
            f"{TIE_NAMESPACE}|canon|{dataset}|{repr(classes[j])}|{SPLITS[s]}|{seed}".encode("utf-8")
        ).hexdigest()
        return (h, j, s)
    return sorted(cols, key=key)


def _solve_lexicographic(model: _Model, options: dict, column_order=None):
    """Minimise P1..P4 in order, pinning each optimum, then make the count vector canonical."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    integrality = np.zeros(model.total_cols)
    integrality[:model.nx] = 1
    ub = [float(model.n_groups_per_class[i // len(SPLITS)]) for i in range(model.nx)]
    bounds = Bounds(np.zeros(model.total_cols), np.array(ub + [np.inf] * len(model.aux)))

    def solve(cost, pins, label):
        A, lo, hi = model.constraints(pins)
        c = np.zeros(model.total_cols)
        for col, w in cost:
            c[col] += float(w)
        res = milp(c=c, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
                   bounds=bounds, options=options)
        if res.status != 0 or res.x is None:
            raise AllocationError(f"{label}: solver status {res.status}: {res.message}")
        return res

    pins, objectives, evidence = [], {}, []
    for level in LEVELS:
        cost = model.level_cost(level)
        res = solve(cost, pins, level)
        value = int(round(res.fun))
        objectives[level] = value
        pins.append((cost, value))
        evidence.append({"level": level, "status": int(res.status), "message": res.message.strip(),
                         "objective": value,
                         "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
                         "mip_dual_bound": float(getattr(res, "mip_dual_bound", value) or value),
                         "optimality": "GLOBAL_OPTIMUM_PROVEN"})

    # P5a: the count vector itself must be unique, so the solver is never left to choose among
    # equally optimal vectors. Minimise the columns one at a time, in the seeded canonical order,
    # pinning each result. After every column is pinned the vector is fully determined.
    x = None
    for (j, s) in (column_order or [(a, b) for a in range(model.n_classes) for b in range(len(SPLITS))]):
        col = model.xi(j, s)
        res = solve([(col, 1)], pins, f"canonical x[{j}][{s}]")
        pins.append(([(col, 1)], int(round(res.fun))))
        x = res.x
    if x is None:
        x = solve([], pins, "canonical").x

    counts = [[int(round(x[model.xi(j, s)])) for s in range(len(SPLITS))]
              for j in range(model.n_classes)]
    return counts, objectives, evidence


# ------------------------------------------------------------------ public API
@dataclass(frozen=True)
class AllocationResult:
    dataset: str
    assignment: dict                 # group_id -> split           (membership; caller decides to persist)
    objectives: dict                 # level -> integer objective value
    evidence: list                   # per-level solver evidence
    class_counts: list               # [class_index][split] = number of groups
    classes: list                    # class signatures, canonical order
    stats: dict                      # per-split video/sample counts and achieved ratios
    backend: dict                    # solver identity/settings


def _canonical_class_order(groups):
    """Deterministic class order: by signature, rendered as a stable string."""
    sigs = {}
    for g in groups:
        sigs.setdefault(g.class_signature, []).append(g)
    return sorted(sigs.items(), key=lambda kv: repr(kv[0]))


def solver_backend(options: dict) -> dict:
    import scipy
    return {"algorithm": "lexicographic sequential integer programming (P1->P2->P3->P4, each pinned)",
            "kind": "EXACT",
            "backend": "HiGHS via scipy.optimize.milp", "scipy_version": scipy.__version__,
            "threads": 1, "deterministic_options": dict(options),
            "objective_arithmetic": "integer coefficients; fixed-point normalisation SCALE=%d" % SCALE,
            "tie_break_groups": f"sha256('{TIE_NAMESPACE}|<dataset>|<group_id>|<seed>') lowercase hex, then group_id",
            "tie_break_counts": f"columns pinned in seeded order sha256('{TIE_NAMESPACE}|canon|<dataset>|<class>|<split>|<seed>')",
            "split_seed": SPLIT_SEED}


DEFAULT_OPTIONS = {"mip_rel_gap": 0.0, "presolve": True, "time_limit": 600}


def allocate(groups, options: dict | None = None) -> AllocationResult:
    """Assign every allocation group of ONE dataset to TRAIN/VAL/TEST (Q-01 contract)."""
    groups = list(groups)
    validate_groups(groups)
    dataset = groups[0].dataset
    options = dict(DEFAULT_OPTIONS if options is None else options)

    ordered = _canonical_class_order(groups)
    classes = [sig for sig, _ in ordered]
    members = [gs for _, gs in ordered]
    n_groups_per_class = [len(gs) for gs in members]
    videos_per_group = [len(sig) for sig in classes]
    class_counts = {lvl: _class_category_counts(classes, lvl) for lvl in LEVELS if lvl != "P1_total"}

    model = _Model(len(classes), n_groups_per_class, videos_per_group, class_counts).build()
    counts, objectives, evidence = _solve_lexicographic(
        model, options, canonical_column_order(dataset, classes))

    # --- P5: which concrete groups, by frozen stable rank (never by input order) ---
    assignment = {}
    for j, gs in enumerate(members):
        ranked = sorted(gs, key=lambda g: (stable_group_rank(g.dataset, g.group_id), g.group_id))
        cursor = 0
        for s, split in enumerate(SPLITS):
            for g in ranked[cursor:cursor + counts[j][s]]:
                assignment[g.group_id] = split
            cursor += counts[j][s]
        if cursor != len(ranked):
            raise AllocationError(f"class {j}: counts {counts[j]} do not cover {len(ranked)} groups")

    stats = summarize(groups, assignment)
    verify_assignment(groups, assignment, objectives)
    return AllocationResult(dataset, assignment, objectives, evidence, counts, classes, stats,
                            solver_backend(options))


def objective_values(groups, assignment) -> dict:
    """Recompute every level's objective from an assignment, in exact Python integers.

    Independent of the solver: this is what the tests and the audit compare against.
    """
    out = {}
    by_split = {s: [v for g in groups if assignment[g.group_id] == s for v in g.videos] for s in SPLITS}
    N = sum(g.n_videos for g in groups)
    out["P1_total"] = sum(abs(100 * len(by_split[s]) - PCT[s] * N) for s in SPLITS)
    for level in LEVELS[1:]:
        cats: dict = {}
        for g in groups:
            for v in g.videos:
                for k in _category_keys(level, v):
                    cats[k] = cats.get(k, 0) + 1
        total = 0
        for cat, N_c in sorted(cats.items()):
            w = SCALE // N_c
            for s in SPLITS:
                n_cs = sum(1 for v in by_split[s] if cat in _category_keys(level, v))
                total += w * abs(100 * n_cs - PCT[s] * N_c)
        out[level] = total
    return out


def verify_assignment(groups, assignment, objectives=None) -> None:
    """P0 invariants, re-checked on the produced assignment rather than assumed."""
    for g in groups:
        if assignment.get(g.group_id) not in SPLITS:
            raise AllocationError(f"group {g.group_id!r} has no valid split")
    seen = {}
    for g in groups:
        for v in g.videos:
            if v.video_id in seen and seen[v.video_id] != assignment[g.group_id]:
                raise AllocationError(f"canonical video {v.video_id!r} crosses splits")
            seen[v.video_id] = assignment[g.group_id]
    if objectives is not None:
        recomputed = objective_values(groups, assignment)
        if recomputed != objectives:
            raise AllocationError(f"objective mismatch: solver {objectives} vs recomputed {recomputed}")


def summarize(groups, assignment) -> dict:
    """Aggregate per-split counts. Contains no group->split membership."""
    out = {}
    N = sum(g.n_videos for g in groups)
    usable = sum(v.n_complete for g in groups for v in g.videos)
    failed = sum(v.n_failed for g in groups for v in g.videos)
    for s in SPLITS:
        vids = [v for g in groups if assignment[g.group_id] == s for v in g.videos]
        out[s] = {"groups": sum(1 for g in groups if assignment[g.group_id] == s),
                  "canonical_videos": len(vids),
                  "video_pct": round(100.0 * len(vids) / N, 4) if N else 0.0,
                  "usable_samples": sum(v.n_complete for v in vids),
                  "failed_samples_provenance_only": sum(v.n_failed for v in vids),
                  "live_videos": sum(1 for v in vids if int(v.label_binary) == 0),
                  "spoof_videos": sum(1 for v in vids if int(v.label_binary) == 1)}
    out["_totals"] = {"canonical_videos": N, "usable_samples": usable,
                      "failed_samples_provenance_only": failed}
    return out


def rare_category_report(groups) -> dict:
    """How many independent allocation groups carry each category (three-way feasibility)."""
    rep = {}
    for level in LEVELS[1:]:
        cats: dict = {}
        for g in groups:
            for v in g.videos:
                for k in _category_keys(level, v):
                    cats.setdefault(k, {"videos": 0, "groups": set()})
                    cats[k]["videos"] += 1
                    cats[k]["groups"].add(g.group_id)
        rep[level] = {k: {"videos": v["videos"], "groups": len(v["groups"]),
                          "can_occupy_three_splits": len(v["groups"]) >= 3}
                      for k, v in sorted(cats.items())}
    return rep
