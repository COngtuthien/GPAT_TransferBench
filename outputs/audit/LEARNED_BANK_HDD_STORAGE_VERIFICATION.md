# M8 learned-bank HDD storage verification (2026-11-22)

Machine-readable evidence: `outputs/audit/LEARNED_BANK_HDD_STORAGE_VERIFICATION.json`.
Storage authority: `configs/amendments/learned_bank_storage_location.yaml` (OWNER_APPROVED, L1 = B_EXTERNAL_HDD_ADDITIVE_ROOT).

**Scope.** This covered storage setup and verification only. No S7 was run, and there was no generator inference, no bank, no training, no TEST evaluation, no GPAT and no GPU use.

## Physical storage (GPU host nd-System-Product-Name, user student20261)

| Item | Value |
|---|---|
| Device | /dev/sda2 on /dev/sda: Seagate One Touch w/PW, USB, 3.6 TiB |
| Filesystem | ext4, UUID `608647d2-9cb0-45b0-b999-73c080ba909b` (by-uuid points to ../../sda2, dev 8:2) |
| Canonical mount | `/media/ext_data`: per-mount `rw,relatime`, superblock `rw,errors=remount-ro`, mounted once and not shadowed. The mount point is inode 2 and stays root:shared 2775 (unchanged) |
| Duplicate mount | `/media/student20261/608647d2-...` (udisks) was never used |
| Root SSD | /dev/nvme0n1p3 (different device, 97% used). It is not the bank root |
| Free space | 3,450,618,556,416 bytes available. 244,037,512 inodes free |

## Private directory

The directory was **newly created** at 2026-11-22T16:26:37Z, under the owner's explicit authorization. It was created with an exclusive `os.mkdir`, and nothing existed at the path beforehand.

The new directory inherited setgid from the parent, so its mode was 2700 right after mkdir. An `fchmod(0o700)` was then applied to that directory only, through an O_NOFOLLOW fd. Nothing was changed recursively and the parent was not touched.

Final state:

- Path: `/media/ext_data/student20261_gpat_m8_banks`
- Owner and group: student20261:shared (group has no access)
- Mode: exactly **0700**
- Inode: 233832449
- Symlink: no
- Realpath: equal to the path
- Contents: **empty**

The permission probe wrote a 73-byte non-scientific file with O_EXCL. Its SHA256 matched when read back. The probe was then removed. Afterwards the directory was empty and its owner, mode, inode and UUID were unchanged.

## Live preflight (clean clone at d163106, `tools/learned_bank.py storage-preflight`)

All 13 checks passed:

- exists
- not_symlink
- is_directory
- realpath_equals_path
- owner
- mode_0700_no_setgid
- not_udisks_duplicate
- mount_point
- filesystem_uuid
- fstype_ext4
- mount_read_write
- user_rwx_access
- no_unexpected_contents

## Registry

Banks resolve to `<bank_root>/<method>/seed_<seed>/` for E03, E05, E06b and E06c and seeds 42, 1337 and 2026. Resolution goes only through `resolve_registered_bank` and `production_bank_dir`.

The registry accepts only the registered HDD root. It refuses the udisks prefix, relative or non-normalized roots, and traversal. No bank or method directory was created.

## Unchanged

- **E04 seed 2026:** PID 435565 kept running. The GPU checkout is still at 18ba00e and clean. Only `ps`, `tmux ls` and `git rev-parse`/`status` were used on it.
- **E01/E02:** the writer, readers and authority files are unchanged. 140 non-regression tests pass.
- **No reads:** no face or scientific data was read.

## Remaining blockers for banks

- E04 seed-2026 completion and three-seed closure.
- The owner's S7 launch authority.
- S7 acceptance.
- The per-method production authority.

Storage is no longer a blocker.
