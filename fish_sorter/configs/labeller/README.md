# Finding Dory labeller config

`config.json` (sibling to this file) tells Finding Dory where your trained
model checkpoint and DINOv3 assets live. Copy `config.example.json` to
`config.json` and fill in the paths by hand. If `config.json` is missing or
unparseable when you click **Finding Dory**, a popup tells you to create it
and the dock won't open until it exists.

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

Finding Dory does not download anything, and does not create or edit the
config. If a path is wrong it surfaces the error — fix `config.json` by hand.

## Per-mode fields

- `checkpoint_path` — `best.ckpt` from training for this mode.
- `model_arch` — `dinov3_vitb16` (or whatever variant the ckpt was trained
  with). The bare variant (`vitb16`) is also accepted.
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
- `batch_size` — inference mini-batch size for the forward pass. `null`
  (default) uses a per-device default (cuda `32`, mps `16`, cpu `8`). Raise
  it on a GPU with spare VRAM for faster throughput; lower it if you hit
  out-of-memory errors.
- `prewarm_embeddings` — (default `true`) compute embeddings, the per-channel
  UMAP layout, and clusters in the background as soon as the mosaic finishes
  stitching, so the Finding Dory dock opens instantly instead of running the
  model + UMAP fit on click. Pre-warm embeds *every* well; the singlet filter
  (below) is applied when the dock adopts the result. Note the first UMAP fit
  JIT-compiles numba kernels and briefly freezes the GUI — with pre-warm on,
  that cost is paid in the background phase rather than on first dock open. Set
  `false` on slow/CPU-only machines to defer all embedding work until Finding
  Dory is actually opened. The very first run on a fresh machine never
  pre-warms (the config doesn't exist until you finish setup), so that run
  computes on click regardless.
- `filter_to_singlets` — (default `true`) restrict the embedding view to
  wells Finding Nemo flagged as singlets (auto-running `find_fish` if needed).
  Set `false` to embed and show every well, including empties / multiples /
  deformed.
- `dev_mock_embeddings` — (default `false`) skip the model entirely and
  generate well-separated synthetic embeddings per channel. For iterating on
  the dock UI without loading a checkpoint or running the forward pass; leave
  `false` for real use.
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
