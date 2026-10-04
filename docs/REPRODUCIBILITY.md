# Installation and workflow

## OpenVLA / OpenVLA-OFT environment

Use a dedicated Python 3.10 environment. The inherited dependencies in
`pyproject.toml` include PyTorch 2.2.0 and TensorFlow 2.15.0; CUDA compatibility
and GPU memory must be checked for your machine. These instructions have not
been validated by a complete GPU training run for this release.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install LIBERO separately and provide its datasets and initial-state files,
plus a compatible pretrained checkpoint. Dependencies include git-hosted forks;
an internet connection and git are required for installation. Original upstream
package metadata remains in place. Package discovery excludes `pi05/` so its
independent environment is not installed into OpenVLA.

## Research entry points

1. Inspect `pm-scripts/generate_poison_data/poison_generator_fc.py` and its
   command-line arguments for source/target tasks, frame pairing, checkpoint,
   perturbation budget and output paths. Its default PGD setting is 500 steps,
   a 16/255 budget, and a 1/255 step size. Supply actual paths and verify the
   model configuration; historical launchers are examples, not final manifests.
2. Inject images with `pm-scripts/inject_to_hdf5/inject_poison_to_hdf5.py`.
   Set `LIBERO_CLEAN_SPATIAL_DIR` to the input HDF5 suite even when using Object,
   `LIBERO_POISONED_SPATIAL_DIR` to a separate output directory,
   `LIBERO_POISON_PT_DIR` to generator outputs, and `LIBERO_TARGET_TASK_FILE`
   to `pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5` for Object.
   Keep `LIBERO_VISUAL_ONLY_MODE=true`, `LIBERO_VERIFY_ACTIONS_UNCHANGED=true`
   and `LIBERO_POISON_SOURCE_LABEL_FRACTION=0.0` for the manuscript threat model.
   The default demo/frame fractions are 0.2/1.0; verify against your intended run.
3. Inspect `vla-scripts/finetune.py` for adaptation options and the selected
   action interface; train using the injected dataset in a separate output path.
4. Evaluate with `experiments/robot/libero/run_libero_eval_dual.py`. Set source
   and target instructions explicitly. Automatic source reward is not a manual
   target-completion score. Review [the scoring protocol](EVALUATION.md) before
   processing any legacy annotation.

Use `--help` for argument-based Python entry points. The injector is configured
through environment variables, not an argparse interface. Do not assume saved
images correspond to the same pre-update MSE evaluation: the retained optimizer
selects an updated image using an earlier recorded loss, as documented in the
manuscript. No optimizer, pairing or scoring implementation was changed here.

## pi0.5 environment

Use a different Python 3.11+ environment and follow
[the integration guide](../pi05/CLOAKVLA_INTEGRATION.md).

```bash
cd pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`uv` must already be installed. Supply LIBERO and a compatible base checkpoint.
Inspect `pm-scripts/finetune_object.sh` before use; its paths can be configured
through `LOCAL_PI05_BASE` and `HF_LEROBOT_HOME`. Evaluation launchers also require
`CKPT_DIR`, `LIBERO_DATASET_PATH` and `POISON_METADATA_PATH`. Their historical
`third_party/libero` path is not bundled; install LIBERO separately and configure
its import path. Verify that poison metadata selects salad dressing for Object.

These files come from the recorded OpenPI snapshot, not a verified latest
uncommitted server working tree. No pi0.5-specific poison generator or physical
Piper deployment is included. Do not claim full paper reproduction from syntax
checks or assume that automatic source failure establishes target completion.
