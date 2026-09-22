# M6D0c — controlled GPU repository synchronization

**COMPLETED. GPU REPOSITORY SYNCHRONIZED EXACTLY TO 4bc78bb. FAST-FORWARD ONLY.**

Recorded UTC: `2026-09-22T17:38:30.231001+00:00`. This is an operational synchronization milestone.
It grants no method execution or environment readiness approval. Historical
M6D0/M6D0b reports and the environment live addendum remain unchanged.

## Laptop and SSH gates

Laptop starting HEAD and origin/m6-baselines: `4bc78bbabedc73e58d24874097506b28d50b0390`.
Branch `m6-baselines`, divergence `0 0`, initially clean. Public-key SSH passed
with BatchMode=yes, PreferredAuthentications=publickey, PasswordAuthentication=no.
Verified host `nd-System-Product-Name`, user `student20261`.
The sandbox initially denied the network socket; approved network escalation
then succeeded. No private-key contents were inspected.

## Controlled GPU operation

Repository: `/home/student20261/workdir/GPAT_TransferBench`.
Pre-sync HEAD: `89ea43e3d32d87d0641ec57bb62e6c72694dbe8a`, branch `m6-baselines`, clean.
`git remote -v` and `git log -5 --oneline --decorate` were captured in JSON.
`git ls-remote origin refs/heads/m6-baselines` returned exactly `4bc78bbabedc73e58d24874097506b28d50b0390`.

Exactly one fetch ran successfully:

```sh
git fetch --no-tags origin refs/heads/m6-baselines:refs/remotes/origin/m6-baselines
```

After fetch, checked-out HEAD remained `89ea43e3d32d87d0641ec57bb62e6c72694dbe8a` and origin/m6-baselines
was `4bc78bbabedc73e58d24874097506b28d50b0390`. `git merge-base --is-ancestor` returned exit 0:
**FAST_FORWARD** ancestry, exactly **16 commits**. The captured diff stat is
148 files changed, 29569 insertions, 144 deletions. These changes already exist
in the authorized target commit. The earlier M6D0b audit used target a6f78e9
and recorded 15 commits; its own additional commit accounts for the distance 16.

The only checked-out GPU repository mutation was:

```sh
git merge --ff-only origin/m6-baselines
```

Git returned exit 0 and `Updating 89ea43e..4bc78bb`, followed by `Fast-forward`.
Immediate exactness gates passed. Post-sync HEAD and origin/m6-baselines both
are `4bc78bbabedc73e58d24874097506b28d50b0390`; branch `m6-baselines`, divergence `0 0`, worktree clean.
The reflog independently reports `merge origin/m6-baselines: Fast-forward`.
No merge commit was created. No fetch or merge was repeated.

The tool transport truncated the long merge file listing, which is preserved
as a truncated transcript in JSON. The successful wrapper enforced all immediate
post-sync gates; subsequent read-only Git queries and reflog corroborate them.
Pre-fetch, post-fetch, diff-stat, hash and runtime outputs are retained in JSON.

## Byte identity and runtime

All ten prescribed files match byte-for-byte by SHA256:

| Path | Laptop and GPU SHA256 | Result |
| --- | --- | --- |
| `configs/methods/e01_fas_aug.yaml` | `87012ea73ab4195ae01898d2be9558b95713e2dca4cc340b5e38cf8936dbc9e0` | MATCH |
| `configs/methods/e03_stdn.yaml` | `08dde851bc9e8ac688c794a6a6999204a7e5b75fb244fcd97e46b22dcb302fe0` | MATCH |
| `configs/methods/e04_physics_std.yaml` | `418617942ebd87c45cb92bb7dc9ad96a6500a104489a657f9ca3345c3686576a` | MATCH |
| `configs/methods/e05_pcgan.yaml` | `478756e150c427832315800acba954cedf778bac520eee3bde8cabf7359e71de` | MATCH |
| `configs/methods/e06c_dsdg_bin_idfree.yaml` | `7176dd4cd49007320ab0c44519e7efde60568098503077303218bd913b6b5002` | MATCH |
| `configs/methods/e07c_difffas_bin_idfree.yaml` | `dba34a9222b80ed66a73b8662c10cd348ab46e4f00c2e8166d0a4fe6d552bc1c` | MATCH |
| `configs/run_logging_v1.yaml` | `d33622947d1951d71bbda4cf86b96c6500943d354fe607beb6cbc47f51c3de15` | MATCH |
| `environments/m6_environment_live_addendum.md` | `e4e9096ee54867003c184ecff8443ea038a95566cd19849a456806ea4dbffee4` | MATCH |
| `outputs/audit/M6D0B_LIVE_GPU_VERIFICATION.json` | `2cf729a3f32eed1cd4a8f10837d0c3fe286b57461c2071b959839a1db17505bb` | MATCH |
| `tools/m6d0b_live_gpu_preflight.py` | `001a5edf08d7d85b86e5625660ea636e49b4de65f92318fe95c4a5c059e94394` | MATCH |

Runtime root `/home/student20261/workdir/GPAT_TransferBench_runtime` and its
`data/processed/faces_256` directory both exist. PNG filename metadata count:
**20615**. Only directory tests and `find ... -type f -name '*.png' -printf '.' | wc -c`
were used. No runtime write, dataset hashing, or pixel decoding occurred.

E04 3DDFA assets were **NOT deployed**. E06c LightCNN was **NOT deployed**.
E07c auxiliary encoder was **NOT created** and remains **NOT_YET_TRAINED**.
M6D0b missing-asset conclusions remain unresolved; no new asset inventory or
scientific/environment question was re-resolved. Environment mutation = **false**.

## Append-only ledger and final validation

Committed ledger: **96 rows**, **241669 bytes**, full-prefix SHA256
`a3d8a59c8ba3eea4c7ff327dea99313e99c2b52428ba1713a67a8bec84293717`, calculated before modification.
Exactly one `M6D0C_GPU_REPOSITORY_SYNC` row is appended: **97 final rows**.
The first 96 rows remain byte-identical. The appended row binds the three new
artifact hashes; the index and ledger are excluded from these bindings to avoid
hash cycles. All five final hashes are reported after final verification.

The validation tool performs local read-only evidence checks; it does not
connect to the GPU or synchronize repositories. Validation sequence:

```sh
python tools/m6d0c_gpu_repo_sync_preflight.py
python tools/build_artifact_index.py
python tools/m6d0c_gpu_repo_sync_preflight.py --final
git diff --check
```

The index builder is the last file mutation, after the new artifacts and ledger
are final. The final validator checks **468 unique paths**, current sizes and
SHA256 values, CRLF preservation, complete index coverage, ledger prefix identity,
and the exact five-file firewall. Index hashing reads tracked files as opaque
bytes, including manifests; it does not interpret them or open referenced images.
Python bytecode writes are disabled for validation and index building.
The only laptop changes are the three M6D0c additions and ledger/index updates.

## Explicit safety declarations

GPU REPOSITORY SYNCHRONIZED EXACTLY TO 4bc78bb; FAST-FORWARD ONLY.
NO SCIENTIFIC CONTENT EDIT; NO RUNTIME WRITE; NO PACKAGE INSTALL;
NO ENV CREATION; NO ENV MUTATION; NO ASSET DEPLOYMENT;
NO MODEL CONSTRUCTION; NO WEIGHT LOAD; NO BENCHMARK IMAGE DECODING;
NO TRAINING; NO INFERENCE; NO SAMPLING; NO SYNTHETIC BANK; NO CHECKPOINT;
NO TEST EXECUTION; NO LAPTOP COMMIT; NO LAPTOP PUSH.
No optimizer, backward pass, GPU commit, manual GPU repository edit, or environment
lock creation occurred. No pull, reset, force checkout, rebase, cherry-pick, or
force push was used. GPU files changed solely through the authorized fast-forward
of existing committed content; laptop scientific files were not edited.
