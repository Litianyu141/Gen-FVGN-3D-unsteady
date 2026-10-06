# Label-Free Finite-Volume-Residual Training of Attention Graph Neural Networks for Coupled Thermo-Fluid Fields

Source code of the paper [arXiv:2607.20321](https://arxiv.org/abs/2607.20321): an attention graph neural network (TransFVGN) trained with the finite-volume method (FVM) residuals of the governing equations on 3D unstructured meshes, together with the data-supervised baselines.

## Layout

```
src/
  entries/       training, rollout and label-conversion entry points
  FVsolver/      differentiable FVM residuals
  FVdomain/      mesh loading, memory-mapped state pool, graph pipeline, boundary conditions
  NNmodels/      TransFVGN, attention-free GNN, Transolver
  Pipeline/      FVM-residual and supervised objectives
  Diagnostics/   rollout errors, force coefficients, step timing
  Post_process/  VTU/VTM export and slice rendering
  Utils/         command-line parameters, optimizers, logging
```

## Installation

```bash
conda create -n FVGN-pt2.10 python=3.13 -y
conda activate FVGN-pt2.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # PyTorch 2.10, CUDA 12.8
pip install torch_geometric
pip install -r requirements.txt
```

Only PyTorch Geometric's pure-Python functionality is used, so `torch_scatter`, `torch_sparse` and `pyg_lib` are optional.

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
