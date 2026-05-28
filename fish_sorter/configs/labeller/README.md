# Finding Dory labeller config

`config.json` (sibling to this file) tells Finding Dory where your trained
model checkpoint and DINOv3 assets live. The first time you click **Finding
Dory** with no `config.json` present, a setup dialog opens and creates one
for you via file pickers (checkpoint + DINOv3 repo + mode). You can also copy
`config.example.json` to `config.json` and fill in the paths by hand.

## What you need before launching Finding Dory

1. **A trained model checkpoint (`best.ckpt`)** for each mode (egg / fish)
   you plan to use. These come from the zebra-classify training pipeline.
2. **The DINOv3 git repo** cloned locally (needed by `torch.hub.load` to
   construct the ViT class — it loads Python source, not weights, so the
   clone is small).
3. (**Optional**) **The DINOv3 pretrained weights** (e.g. `dinov3_vits16.pth`).
   These are only used as a fallback for keys your trained checkpoint
   doesn't cover. A normal BYOL checkpoint covers the entire backbone, so
   you can leave `dinov3_weights_dir: null` in the config and skip this
   download. Watch the load log: if `missing_keys` is large (hundreds, not
   <10), point at a weights dir as a backstop.

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
- `prewarm_embeddings` — (default `true`) compute embeddings in the
  background as soon as the mosaic finishes stitching, so the Finding Dory
  dock opens instantly instead of running the model on click. Pre-warm
  embeds *every* well; the singlet filter (below) is applied when the dock
  adopts the result. Set `false` on slow/CPU-only machines to defer all
  embedding work until Finding Dory is actually opened. The very first run on
  a fresh machine never pre-warms (the config doesn't exist until you finish
  setup), so that run computes on click regardless.
- `filter_to_singlets` — (default `true`) restrict the embedding view to
  wells Finding Nemo flagged as singlets (auto-running `find_fish` if needed).
  Set `false` to embed and show every well, including empties / multiples /
  deformed.
- `pick_type_to_mode` — maps fish-sorter `pick_type` values (the dropdown
  in the Setup tab) to model bundles.
- `clustering.method` / `params` — selects the cluster strategy
  (`hdbscan` is the only one shipped today).
- `umap.n_neighbors` / `umap.min_dist` — UMAP layout params. Higher
  `min_dist` spreads points more evenly within clusters (less clumping),
  which also lets the Show-Fish overlay show full-resolution thumbnails
  with less overlap at a given canvas size.
- `umap.canvas_px` — (default `8192`) the maximum side length, in pixels,
  of the "Fish UMAP" composite image (the Show-Fish overlay). Fish
  thumbnails are always drawn at full resolution; this only bounds how
  large the backing image can get. In dense clusters a smaller value packs
  fish closer (more overlap) but renders far faster and uses far less
  memory; a larger value separates them more at the cost of render time.
  The previous hardcoded value (32000) could produce multi-gigabyte
  textures and multi-minute renders on busy plates.
