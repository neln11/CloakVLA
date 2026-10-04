# CloakVLA pi0.5 integration

This directory preserves the source snapshot of `neln11/openpi0.5` at commit
`9351a19e41c5795c0a216deb6d12150d8fd35282`. It contains the OpenPI model,
training and serving code, OpenPI client, LIBERO conversion/evaluation code,
and the author's LIBERO poisoned-dataset training/evaluation launchers.
Copied source files are unchanged; their checksums are in SOURCE_MANIFEST.json.

## Included research entry points

- `pm-scripts/finetune_{spatial,object,goal}.sh`
- `pm-scripts/eval_dual_{spatial,object,goal,all}.sh`
- `src/openpi/training/config.py`: `pi05_libero_*_poisoned_no_noops` configurations.
- `examples/libero/eval_libero_dual.py`: ordinary-suite and source-control/trigger evaluation.
- `examples/libero/convert_libero_data_to_lerobot.py`: dataset conversion.

## Environment

Use an independent Python 3.11+ environment for this subtree. OpenVLA and
OpenPI use different dependencies; do not merge their environments.
From this directory, follow the preserved README and pyproject.toml for `uv`
installation and dependency synchronization. Supply LIBERO, the chosen dataset,
and model weights separately. Inspect each launcher before running it.

The launchers/configuration contain historical `/data1/zhaoxueyang_191/` paths.
They must be configured for your installation. They are not portable defaults.
Do not interpret these paths as bundled model weights or data.

## Evaluation and release scope

The automatic failure/disruption measure in the dual evaluator is based on
source-task reward. It does not automatically measure completed target behavior.
The paper's target-behavior score requires separate manual review using
`../docs/EVALUATION.md`; `../scripts/summarize_target_scores.py` validates and
summarizes completion-based annotations without modifying them.

This snapshot does not contain a pi0.5-specific feature-collision/PGD poison
generator, historical manual annotations, model weights, datasets, or Piper
deployment code. It is not a complete reproduction archive of every paper
experiment. The server's unpushed working-tree changes have not been verified.

## License

OpenPI code in this subtree retains its Apache-2.0 LICENSE and LICENSE_GEMMA.txt.
The root MIT license for the OpenVLA-derived subtree does not replace these
licenses. No third-party model weights are redistributed.
