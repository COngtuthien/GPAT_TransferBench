# GPU Connectivity Audit — M0 (read-only)

Checked 2026-09-18 from cong-ThinkBook-16-G7-AHP. No project was created, copied, synced
or installed on the GPU server.

| Check | Result |
|---|---|
| SSH alias `sparc5090` in `~/.ssh/config` | PRESENT → HostName `server-node.wondertek.space`, User `sparc`, Port `2426` (same as fallback) |
| DNS `server-node.wondertek.space` | resolves to 210.245.70.125 (CNAME-like alias `hec08rwqwte.sn.mynetname.net`) |
| TCP port 2426 | OPEN (bash `/dev/tcp` probe, 8 s timeout) |
| Private keys in `~/.ssh` | NONE (only `config`, `authorized_keys`, `known_hosts*`) |
| ssh-agent identities | none ("The agent has no identities.") |
| Non-interactive login `ssh -o BatchMode=yes -o ConnectTimeout=10 sparc5090 hostname` | **FAILED**, exit 255: `Permission denied (publickey,password).` |

## Status: PENDING

Login requires an interactive password. Per M0 instructions the GPU audit stopped here.
GPU model, driver, CUDA, disk free and Python on the server are **unknown**.
The old GPU project `/home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project` and the planned
root `/home/sparc/workdir/longnm/GPAT_TransferBench` were **not** inspected.

## To complete later (owner action)

Either run the read-only commands interactively, e.g. in this session:

```
! ssh sparc5090 'hostname; nvidia-smi; df -h /home/sparc/workdir; python3 --version; ls -ld /home/sparc/workdir/longnm/PRISM_FAS_C_LLM_Project /home/sparc/workdir/longnm/GPAT_TransferBench'
```

or install a key for non-interactive access (owner decision; not done here).
