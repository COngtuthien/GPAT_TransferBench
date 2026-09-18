# M2A Environment Report (2026-09-19)

| Item | Value |
|---|---|
| Env | `/home/cong/.venvs/gpatbench-m2` (isolated; outside the project root; the M1 `.venv` is untouched) |
| Lock | `environments/m2_preprocess_laptop.lock.txt` (pip freeze --all; sha256 in `ARTIFACT_INDEX.csv`) |
| Runtime facts | `environments/m2_preprocess_laptop.runtime.json` |
| Python | 3.12.3 |
| PyTorch / torchvision | 2.14.0+cpu / 0.29.0+cpu (index https://download.pytorch.org/whl/cpu) |
| CUDA / cuDNN | not available (`torch.cuda.is_available() == False`; laptop GPU is an AMD Phoenix3 iGPU, no NVIDIA) |
| ONNX Runtime | 1.30.0, providers [Azure, CPU]; CPU used |
| ONNX | 1.22.0 (graph inspection only) |
| OpenCV | opencv-python-headless 5.0.0.93 (FFmpeg avcodec 62.28.101 / avformat 62.12.101 / avutil 60.26.101 / swscale 9.5.101), same as M1 |
| numpy / pyarrow / PyYAML / Pillow | 2.5.3 / 25.0.1 / 6.0.3 / 12.3.0 |
| PyAV | not needed for M2 (frames use the M1 OpenCV decoder; PyAV remains the audit reference, `m1_decoder_audit_laptop.lock.txt`) |
| FaceXFormer deps | torch, torchvision (swin_b) only; the official demo's `facenet_pytorch` (MTCNN) is not used |
| AdaFace deps | torch only |
| Smoke device | laptop CPU |
| Expected M2B host | **UNDECIDED**. Measured laptop CPU cost per sample (run3; includes an extra diagnostic AdaFace BGR pass): CASIA 0.59 s, MSU 0.75 s, SiW 0.91 s → dataset-weighted ≈ 4.7 h for 20,640 samples, excluding model load. Storage issue Q-23 applies. A remote GPU host would need its own smoke and lock |
