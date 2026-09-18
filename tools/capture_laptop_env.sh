#!/usr/bin/env bash
# Read-only capture of laptop environment state. No installs, no mutations.
# Usage: bash tools/capture_laptop_env.sh > environments/laptop_environment_initial.txt
sec(){ echo; echo "### $*"; }
echo "# GPAT-TransferBench laptop environment capture"
echo "captured_utc: $(date -u +%FT%TZ)"
sec hostname/user; hostname; whoami
sec os-release; cat /etc/os-release
sec kernel; uname -a
sec cpu; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
sec memory; free -h
sec disk; df -h /home/cong /media/cong/Data 2>&1
sec display_adapters; lspci 2>/dev/null | grep -iE 'vga|3d|display'
sec nvidia-smi; command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi: NOT FOUND"
sec nvcc; command -v nvcc >/dev/null && nvcc --version || echo "nvcc: NOT FOUND"
sec cudnn; ldconfig -p | grep -i cudnn || echo "libcudnn: NOT FOUND in ldconfig"
sec python; command -v python3; python3 --version; python3 -m pip --version 2>&1; ls /usr/bin/python3* 2>&1
sec conda_mamba_uv; for t in conda mamba micromamba uv; do printf '%s: ' $t; command -v $t || echo "NOT FOUND"; done
sec python_imports
python3 - <<'PY'
import importlib
for m in ["torch","torchvision","cv2","numpy","pandas","pyarrow","onnxruntime","scipy","sklearn","ptwt","pywt","transformers","timm","yaml","pytest","PIL","lpips","insightface"]:
    try:
        x = importlib.import_module(m); print(f"{m}\tAVAILABLE\t{getattr(x,'__version__','?')}\t{getattr(x,'__file__','')}")
    except Exception as e:
        print(f"{m}\tMISSING\t{type(e).__name__}")
PY
sec dpkg_python_and_tools
dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null | grep -E '^(python3|python3\.12.*|python3-(numpy|opencv|yaml|pil|pip-whl|setuptools-whl)|git|openssh-client|rsync|docker.*|containerd.*)\b'
sec git; git --version; echo "global config:"; git config --global --list 2>&1 | grep -viE 'token|password|credential' ; echo "system config:"; git config --system --list 2>&1
sec ssh_rsync; ssh -V 2>&1; rsync --version | head -1
sec containers; for t in docker podman; do printf '%s: ' $t; command -v $t || echo "NOT FOUND"; done
sec ffmpeg; command -v ffmpeg || echo "ffmpeg: NOT FOUND"
