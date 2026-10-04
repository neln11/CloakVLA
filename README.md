# CloakVLA

Research code for clean-label poisoning of vision-language-action policies.

Authors: Xueyang Zhao and Jinyin Chen, Zhejiang University of Technology.
Corresponding author: chenjinyin@zjut.edu.cn.

## Release contents

This source-only preparation contains feature-collision image optimization,
HDF5 image injection, dataset conversion launchers, OpenVLA/OpenVLA-OFT
adaptation code, and LIBERO evaluation utilities. It does not include model
weights, demonstration datasets, rollout videos, manual annotations, or
experiment logs. It is not a complete reproduction archive of every result
in the manuscript. The pi0.5 training/evaluation snapshot is included under `pi05/`; its dedicated
poison generator and Piper deployment code are not included.

## pi0.5

See [pi0.5 integration](pi05/CLOAKVLA_INTEGRATION.md). This Apache-2.0 subtree
uses its own Python 3.11+ environment and preserves the author's existing
OpenPI training/evaluation snapshot.

## Setup and entry points

Use a separate Python environment. Inspect `pyproject.toml` for the inherited
OpenVLA-OFT dependency versions, then install with `pip install -e .`.
LIBERO, datasets, and pretrained checkpoints must be supplied separately.

- Image optimization: `pm-scripts/generate_poison_data/poison_generator_fc.py`
- HDF5 injection: `pm-scripts/inject_to_hdf5/inject_poison_to_hdf5.py`
- Fine-tuning: `vla-scripts/finetune.py`
- Evaluation: `experiments/robot/libero/run_libero_eval_dual.py`

Run Python entry points with `--help` to inspect their configuration options.
Shell launchers are retained as historical examples. Their local paths,
source/target task names, and checkpoints require verification before use;
they are not authoritative descriptions of the final manuscript experiments.
In particular, the manuscript's Object attack pair is alphabet soup -> basket
as the source and salad dressing -> basket as the target; some historical
launchers still select tomato sauce.

## Attribution and license

This code builds on OpenVLA and OpenVLA-OFT. The original MIT license and
upstream copyright notices are retained in `LICENSE` and source files.
Inherited package metadata in `pyproject.toml` is retained for provenance.

## Repository and evaluation

Repository: https://github.com/neln11/CloakVLA .

See [evaluation protocol](docs/EVALUATION.md)
for the current rubric and limitations of the retained historical utilities.
See [Object task configuration](configs/object_attack.json) for the manuscript
attack pair. No manuscript data-availability statement is changed by this release.

## Source checks

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow checks Python syntax and the manual-score validator.
These checks do not run model training, simulation, or physical experiments.
