# Contributing

Please open an issue with the entry point, commit, Python/CUDA versions,
configuration, expected behavior and a minimal error trace. Remove credentials,
personal paths and private data before posting logs.

Use a pull request for changes. Keep source-task reward separate from manual
target-behavior scoring; describe any change to an experimental protocol.
Do not upload model weights, datasets, videos or private annotation records.
Preserve upstream licenses and attribution. Run the source checks before a PR:

```bash
python -m unittest discover -s tests -v
```

GPU-dependent results require their own validation; CI only checks syntax and
the read-only score summarizer.
