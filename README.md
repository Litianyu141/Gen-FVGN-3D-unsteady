# Label-Free Finite-Volume-Residual Training of Attention Graph Neural Networks for Coupled Thermo-Fluid Fields

Source code of [arXiv:2607.20321](https://arxiv.org/abs/2607.20321): Warp-accelerated attention graph neural networks (TransFVGN) trained with differentiable finite-volume method (FVM) residuals on 3D unstructured meshes (PCNO), FIFO-windowed direct residual optimization (ODIL), and data-supervised baselines.

## Layout

```
src/
  entries/       PCNO training/rollout, ODIL solver, supervised training/rollout, label conversion
  FVsolver/      Warp differentiable FVM residual solvers and time integrator
  FVdomain/      mesh loading, GPU/mmap state pool, graph pipeline, Warp boundary conditions
  NNmodels/      Warp TransFVGN, attention-free GNN, Transolver, TransolverPlus
  Pipeline/      PCNO, ODIL, and supervised objectives
  TimeWindow/    FIFO space-time window core, residualControl, FIFOAdam, run I/O
  Diagnostics/   rollout errors, force coefficients, step timing
  Post_process/  VTU/VTM export and slice rendering
  Utils/         CLI parameters, batched SOAP optimizer, schedulers, DDP, logging
```

## Installation

Requires a CUDA-capable GPU for NVIDIA Warp kernels.

```bash
conda create -n FVGN-pt2.10 python=3.13 -y
conda activate FVGN-pt2.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -r requirements.txt
```

Only PyTorch Geometric's pure-Python API is used, so `torch_scatter`, `torch_sparse`, and `pyg_lib` are optional.

## Citation

```bibtex
@article{li2026labelfree,
  title         = {Label-Free Finite-Volume-Residual Training of Attention Graph Neural Networks for Coupled Thermo-Fluid Fields},
  author        = {Li, Tianyu and Cao, Zhiwei and Zhang, Qingang and Wang, Ruihang and Song, Binyang and Wen, Yonggang},
  journal       = {arXiv preprint arXiv:2607.20321},
  year          = {2026},
  eprint        = {2607.20321},
  archivePrefix = {arXiv}
}
```

## License

Apache License 2.0; see [LICENSE](LICENSE).
