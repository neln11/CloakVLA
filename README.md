# CloakVLA

[![Source checks](https://github.com/neln11/CloakVLA/actions/workflows/source-checks.yml/badge.svg)](https://github.com/neln11/CloakVLA/actions/workflows/source-checks.yml)

**CloakVLA: Stealthy and targeted clean-label poisoning of vision-language-action policies**

Research code by Xueyang Zhao and Jinyin Chen, Zhejiang University of Technology.
Corresponding author: chenjinyin@zjut.edu.cn.

CloakVLA modifies selected target-task RGB observations while preserving their
instructions, robot states, and actions. Temporary triggered source references
guide feature collision; the deployment trigger is absent from stored poisons.
Joint activation uses a visual trigger and the target instruction.

## Release scope

This is a source-code release, not a complete reproduction archive of every
manuscript result. Model weights, demonstrations, rollout videos, manual
annotations, and experiment logs are not included.

| Component | Location | Included scope |
| --- | --- | --- |
| OpenVLA / OpenVLA-OFT | Root directory | Visual feature optimization, HDF5 injection, adaptation and LIBERO evaluation |
| pi0.5 / OpenPI | `pi05/` | Model, training, serving, client, dataset conversion and LIBERO evaluation snapshot |
| Target-behavior scoring | `docs/EVALUATION.md`, `scripts/summarize_target_scores.py` | Current rubric and read-only annotation validation |
| Physical Piper experiment | Not included | Deployment code and recordings are not released here |

The pi0.5-specific poison generator is not included. Its snapshot provenance
and installation notes are in [pi0.5 integration](pi05/CLOAKVLA_INTEGRATION.md).

## Getting started

```bash
git clone https://github.com/neln11/CloakVLA.git
cd CloakVLA
```

Use separate environments for the OpenVLA root and `pi05/` subtree.
See [installation and workflow](docs/REPRODUCIBILITY.md) before running code.
GPU training requires external checkpoints and separately installed LIBERO.

The manuscript's LIBERO-Object attack pair is:

- Source: `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- Target: `pick_up_the_salad_dressing_and_place_it_in_the_basket`

See [Object configuration](configs/object_attack.json). Historical launchers
are preserved for provenance and may contain local paths or different task
names, including tomato sauce. They are not authoritative configurations for
the final manuscript; verify every task, checkpoint and path before use.

## Evaluation

See [evaluation protocol](docs/EVALUATION.md). Source-task disruption and the
manual target-behavior score are different quantities. The latter is the mean
of completion-based scores in {0, 0.5, 1}, not a binary attack-success rate.
Legacy utilities contain an older rubric; their historical annotations cannot
be assumed to follow the current rubric without reviewing the full rollouts.

```bash
python scripts/summarize_target_scores.py annotations.csv --expected-rollouts 50
python -m unittest discover -s tests -v
```

The CSV must have `episode_id` and `asr_t_score` columns. The summarizer does
not modify records or infer scores from videos. GitHub Actions checks syntax
and this validator; it does not test training or reproduce experimental results.

## Citation and contributions

Software citation metadata is in [CITATION.cff](CITATION.cff). The manuscript
has no published DOI recorded here. See [contributing](CONTRIBUTING.md) for
reporting reproducibility problems and proposing changes.

## License and provenance

OpenVLA-derived code retains the root [MIT license](LICENSE); OpenPI retains
its own [Apache-2.0 license](pi05/LICENSE) and Gemma notices. See [NOTICE](NOTICE)
and source manifests for attribution. Original upstream package identity and
author metadata are retained; no third-party model weights are redistributed.
