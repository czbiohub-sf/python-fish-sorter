# Finding Dory labeller config

`config.json` (sibling to this file) tells Finding Dory where your trained
model checkpoint and DINOv3 assets live. Copy `config.example.json` to
`config.json` and fill in the paths, or — once Chunk 5 ships — let the
first-time-setup dialog create it for you via file pickers.

## What you need before launching Finding Dory

1. **A trained model checkpoint (`best.ckpt`)** for each mode (egg / fish)
   you plan to use. These come from the zebra-classify training pipeline.
2. **The DINOv3 git repo** cloned locally (needed by `torch.hub.load`).
3. **The DINOv3 pretrained weights** (e.g. `dinov3_vits16.pth`) placed in
   a single directory.

Finding Dory does not download anything. If a path is wrong, the GUI will
prompt you to pick the right file.

## Per-mode fields

- `checkpoint_path` — `best.ckpt` from training for this mode.
- `model_arch` — `dinov3_vits16` (or whatever variant the ckpt was trained
  with). The bare variant (`vits16`) is also accepted.
- `crop_size` — `[H, W]` of the model's expected input. Must match what
  training used; check `MODE_DEFAULTS` in the zebra repo if unsure.
- `pooling` — currently always `gem` (CLS + GeM-pooled patch tokens).
- `contrast` — per-channel percentile + tonemap parameters. **These travel
  with the checkpoint and change between model generations.** Use the
  values the ckpt was trained against; the defaults shipped in
  `config.example.json` are the current zebra-repo defaults at time of
  writing and may be wrong for older ckpts.
  - `BF` block: linear stretch, no asinh.
  - `_FLUOR` block: fallback for any non-BF channel.

## Top-level fields

- `dinov3_repo_path` / `dinov3_weights_dir` — paths to the DINOv3 hub repo
  and the directory containing variant `.pth` files.
- `device` — `auto` resolves cuda > mps > cpu. Override with `cuda`,
  `mps`, or `cpu`.
- `pick_type_to_mode` — maps fish-sorter `pick_type` values (the dropdown
  in the Setup tab) to model bundles.
- `clustering.method` / `params` — selects the cluster strategy
  (`hdbscan` is the only one shipped today).
