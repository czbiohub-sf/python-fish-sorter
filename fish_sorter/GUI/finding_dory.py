"""Finding Dory — embedding-driven labelling dock.

`FindingDory(QWidget)` is the napari dock content for the "Finding Dory"
workflow:

  Click Finding Dory -> embeddings compute in background -> UMAP scatter renders
  -> user lassoes wells, assigns named groups, toggles lHead per well, saves
  CSV.

It shares the active `Classify` instance (and therefore the napari viewer,
points layer, and well coordinates) so it never duplicates the well-extraction
work Finding Nemo already does. On Save it writes a wide CSV that is a strict
superset of Finding Nemo's output, so downstream `SelectGUI` reads either
interchangeably.

The wide-CSV writer at module scope (`write_wide_csv`, `default_csv_path`) is
used both by this dock's Save button and by tests in `tests/test_wide_csv.py`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from fish_sorter.helpers.labelling.store import GLOBAL_GROUPS, LabelStore

log = logging.getLogger(__name__)

# umap-learn prints a TensorFlow warning at import advertising the parametric
# UMAP feature, which we don't use. Silence it so the dock log stays focused
# on actionable messages.
warnings.filterwarnings(
    "ignore", message=".*tensorflow.*parametric.*", module=r"umap.*"
)
warnings.filterwarnings(
    "ignore", message=".*Tensorflow not installed.*", module=r"umap.*"
)

# UMAP's first call triggers Numba JIT compilation, which dumps thousands
# of SSA / bytecode DEBUG lines if the host configured the root logger at
# DEBUG. Cap the noisiest libraries at WARNING so embedding runs stay
# legible. Anything genuinely broken still surfaces.
for _noisy in ("numba", "numba.core", "numba.core.ssa", "numba.core.byteflow",
               "umap", "umap.umap_", "llvmlite"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Wide-CSV serializer (used by Save + by tests)
# ---------------------------------------------------------------------------


def default_csv_path(expt_dir: str, prefix: str, timestamp: Optional[str] = None) -> str:
    """Construct the `{TIMESTAMP}_{PREFIX}_classifications.csv` path.

    Matches `classify.py:316–317` — Finding Dory writes alongside Finding Nemo
    so downstream `SelectGUI` reads either interchangeably.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{timestamp}_{prefix}_classifications.csv"
    return os.path.normpath(os.path.join(expt_dir, fname))


def write_wide_csv(
    store: LabelStore,
    well_order: List[str],
    lhead_map: Dict[str, bool],
    channels: Iterable[str],
    fish_line: str,
    path: str,
    infer_singlet: bool = True,
    well_defaults: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Write a classify-compatible wide CSV from a `LabelStore` snapshot.

    Schema (one row per well_id, columns in this order):

        well_name, empty, singlet, multiple, deformed, lHead,
        {channel0}_{custom_group0}, {channel0}_{custom_group1}, ...,
        {channel1}_{custom_group0}, ...

    Sources for the default columns:
      - If `well_defaults[well_id]` provides a column (`empty`, `singlet`,
        `multiple`, `deformed`, `lHead`), that wins. This is how Finding Dory
        passes Finding Nemo's `points_layer.features` through.
      - Otherwise:
          * `empty`/`multiple`/`deformed` derive from LabelStore global-group
            assignments (legacy default).
          * `singlet`, when `infer_singlet=True`, is the complement of the
            three globals.
          * `lHead` comes from `lhead_map`.

    All scoring columns are int (0/1).
    """
    channels = list(channels)
    well_defaults = well_defaults or {}

    custom_by_channel: Dict[str, List[str]] = {}
    for ch in channels:
        all_groups = store.groups(fish_line, ch)
        custom_by_channel[ch] = [g for g in all_groups if g not in GLOBAL_GROUPS]

    well_name_by_id: Dict[str, str] = {}
    if "well_id" in store.well_metadata.columns and "well_name" in store.well_metadata.columns:
        for _, row in store.well_metadata[["well_id", "well_name"]].iterrows():
            well_name_by_id[str(row["well_id"])] = str(row["well_name"])

    rows = []
    for wid in well_order:
        ext = well_defaults.get(wid, {})

        if "empty" in ext or "multiple" in ext or "deformed" in ext:
            empty_v = int(bool(ext.get("empty", 0)))
            multiple_v = int(bool(ext.get("multiple", 0)))
            deformed_v = int(bool(ext.get("deformed", 0)))
        else:
            # Legacy fallback: derive from LabelStore globals.
            store_flags = {g: 0 for g in ("empty", "multiple", "deformed")}
            for ch in channels:
                assigned = store.assignments(fish_line, ch).get(wid)
                if assigned in store_flags:
                    store_flags[assigned] = 1
            empty_v = store_flags["empty"]
            multiple_v = store_flags["multiple"]
            deformed_v = store_flags["deformed"]

        if "singlet" in ext:
            singlet_v = int(bool(ext["singlet"]))
        elif infer_singlet:
            singlet_v = int(not (empty_v or multiple_v or deformed_v))
        else:
            singlet_v = 0

        if "lHead" in ext:
            lhead_v = int(bool(ext["lHead"]))
        else:
            lhead_v = int(bool(lhead_map.get(wid, False)))

        row = {
            "well_name": well_name_by_id.get(wid, wid),
            "empty": empty_v,
            "singlet": singlet_v,
            "multiple": multiple_v,
            "deformed": deformed_v,
            "lHead": lhead_v,
        }

        for ch in channels:
            assigned = store.assignments(fish_line, ch).get(wid)
            for g in custom_by_channel[ch]:
                row[f"{ch}_{g}"] = int(assigned == g)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"wrote wide CSV {df.shape[0]} rows x {df.shape[1]} cols -> {path}")
    return path


def _subset_embeddings(
    embeds: Dict[str, np.ndarray],
    idx: Dict[str, np.ndarray],
    keep_indices: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Restrict per-channel embeddings to wells in `keep_indices`.

    Used when adopting a background pre-warm (which embeds *all* wells) but the
    `filter_to_singlets` option asks Finding Dory to show only singlets. Rows
    whose original well index isn't in `keep_indices` are dropped from each
    channel's embedding + index arrays.
    """
    keep_set = {int(k) for k in np.asarray(keep_indices).ravel()}
    out_e: Dict[str, np.ndarray] = {}
    out_i: Dict[str, np.ndarray] = {}
    for ch, well_idx in idx.items():
        mask = np.array([int(w) in keep_set for w in well_idx], dtype=bool)
        out_e[ch] = embeds[ch][mask]
        out_i[ch] = well_idx[mask]
    return out_e, out_i


# ---------------------------------------------------------------------------
# First-time setup dialog
# ---------------------------------------------------------------------------


def ensure_labeller_config(cfg_dir: Path, parent=None) -> bool:
    """Ensure `<cfg_dir>/labeller/config.json` exists and is valid.

    If missing or unreadable, opens a GUI dialog with file pickers for
    checkpoint + DINOv3 repo and a mode dropdown, then writes a fresh
    `config.json` derived from `config.example.json` with the chosen paths.

    Returns:
        True if config.json now exists and parses; False if the user
        cancelled or the dialog failed.
    """
    cfg_path = Path(cfg_dir) / "labeller" / "config.json"
    example_path = Path(cfg_dir) / "labeller" / "config.example.json"

    # Already valid? Skip the dialog entirely.
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                json.load(f)
            return True
        except Exception:
            log.warning(f"{cfg_path} exists but failed to parse — running setup.")

    if not example_path.exists():
        log.error(f"missing template: {example_path}")
        return False

    # Lazy imports so non-GUI users (tests, scripts) don't pull Qt.
    from qtpy.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
    )

    class _SetupDialog(QDialog):
        def __init__(self):
            super().__init__(parent)
            self.setWindowTitle("Set up Finding Dory")
            self.setMinimumWidth(560)

            v = QVBoxLayout(self)
            v.addWidget(QLabel(
                "<b>Finding Dory first-time setup.</b><br>"
                "Tell us where your model checkpoint and DINOv3 repo live. "
                "These paths are stored in the labeller config; you'll only "
                "do this once per machine."
            ))

            form = QFormLayout()

            self.ckpt_edit = QLineEdit()
            self.ckpt_btn = QPushButton("Browse…")
            self.ckpt_btn.clicked.connect(self._pick_ckpt)
            ckpt_row = QHBoxLayout()
            ckpt_row.addWidget(self.ckpt_edit, 1)
            ckpt_row.addWidget(self.ckpt_btn)
            form.addRow("Model checkpoint (.ckpt):", _wrap_row(ckpt_row))

            self.repo_edit = QLineEdit()
            self.repo_btn = QPushButton("Browse…")
            self.repo_btn.clicked.connect(self._pick_repo)
            repo_row = QHBoxLayout()
            repo_row.addWidget(self.repo_edit, 1)
            repo_row.addWidget(self.repo_btn)
            form.addRow("DINOv3 repo (folder):", _wrap_row(repo_row))

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["fish", "egg"])
            form.addRow("Mode (which model bundle to use):", self.mode_combo)

            v.addLayout(form)

            self.buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel
            )
            self.buttons.accepted.connect(self._accept)
            self.buttons.rejected.connect(self.reject)
            v.addWidget(self.buttons)

        def _pick_ckpt(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Pick model checkpoint", filter="Checkpoints (*.ckpt);;All (*)"
            )
            if path:
                self.ckpt_edit.setText(path)

        def _pick_repo(self):
            path = QFileDialog.getExistingDirectory(self, "Pick DINOv3 repo folder")
            if path:
                self.repo_edit.setText(path)

        def _accept(self):
            ckpt = self.ckpt_edit.text().strip()
            repo = self.repo_edit.text().strip()
            if not ckpt or not Path(ckpt).exists():
                QMessageBox.warning(
                    self, "Pick a real checkpoint",
                    f"The checkpoint path is missing or doesn't exist:\n{ckpt}",
                )
                return
            if not repo or not Path(repo).exists():
                QMessageBox.warning(
                    self, "Pick a real folder",
                    f"The DINOv3 repo folder is missing or doesn't exist:\n{repo}",
                )
                return
            self._ckpt = ckpt
            self._repo = repo
            self._mode = self.mode_combo.currentText()
            self.accept()

    dlg = _SetupDialog()
    if dlg.exec_() != dlg.Accepted:
        return False

    # Compose the new config from the example template.
    with open(example_path) as f:
        cfg = json.load(f)
    cfg["dinov3_repo_path"] = dlg._repo
    if dlg._mode in cfg.get("models", {}):
        cfg["models"][dlg._mode]["checkpoint_path"] = dlg._ckpt
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info(f"wrote initial labeller config to {cfg_path}")
    return True


def _wrap_row(layout):
    """Wrap a QLayout in a transparent QWidget so it can sit inside a QFormLayout."""
    from qtpy.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w


# ---------------------------------------------------------------------------
# FindingDory dock widget
# ---------------------------------------------------------------------------


def _build_finding_dory():
    """Defer heavy imports (qtpy, matplotlib, napari, umap) until first use.

    Importing this module shouldn't drag in Qt or matplotlib — that lets the
    tiny module-level wide-CSV writer + tests stay cheap.
    """
    import napari
    from qtpy.QtCore import Qt, QTimer, Signal
    from qtpy.QtWidgets import (
        QLabel,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    from fish_sorter.helpers.embedding.clustering import build_cluster_strategy
    from fish_sorter.helpers.embedding.extractor import (
        EmbeddingExtractor,
        compute_embeddings,
        load_config,
        resolve_mode,
    )
    from fish_sorter.helpers.labelling.fish_line import parse_fish_line
    from fish_sorter.helpers.labelling.label_tool import LabelTool as _LabelToolFactory

    class FindingDory(QWidget):
        """Embedding-driven labelling dock — thin wrapper around `LabelTool`.

        Construction kicks off a background thread that builds the embedding
        extractor and runs the forward pass over all wells. When that finishes
        we drop the vendored `LabelTool` widget into the dock as the main
        content; the wrapper here keeps a status panel up top and a Save
        button at the bottom, plus owns the (one-time) lifecycle.
        """

        # Signals — emitted from worker thread, delivered on GUI thread.
        progress_signal = Signal(int, int)
        status_signal = Signal(str)
        embed_done_signal = Signal()
        embed_failed_signal = Signal(str)

        def __init__(self, cfg_dir: Path, classify, parent=None):
            super().__init__(parent)
            self.classify = classify
            self.viewer = classify.viewer
            self.iplate = classify.iplate
            self.prefix = classify.prefix
            self.expt_dir = classify.expt_dir
            self.pick_type = getattr(classify, "picking", "fish")

            # Labeller config + mode resolution.
            cfg_path = Path(cfg_dir) / "labeller" / "config.json"
            self.cfg = load_config(cfg_path)
            self.mode = resolve_mode(self.cfg, self.pick_type)

            # Channels = current Image layers in the viewer.
            self.channels: List[str] = [
                l.name for l in self.viewer.layers if isinstance(l, napari.layers.Image)
            ]
            if not self.channels:
                raise RuntimeError(
                    "Finding Dory needs at least one Image layer in the viewer "
                    "(stitch the mosaic first)."
                )

            # Well metadata for the LabelStore.
            well_names = list(self.iplate.wells["names"])
            self.well_names = well_names
            self.well_ids = [f"{self.prefix}_{n}" for n in well_names]
            self.fish_line = parse_fish_line(self.prefix) or self.cfg.get(
                "fish_line_fallback", "unknown"
            )

            metadata = pd.DataFrame({
                "well_id": self.well_ids,
                "experiment": [self.prefix] * len(self.well_ids),
                "well_name": well_names,
            })
            self.store = LabelStore(metadata)
            self.store._line_channels[self.fish_line] = list(self.channels)
            for ch in self.channels:
                self.store._get_scope(f"{self.fish_line}|{ch}")

            # The LabelStore is seeded with Finding Nemo's global flags later,
            # inside ``_on_embed_done`` — by that point ``_start_embedding``
            # has auto-run ``find_fish`` so ``feat["singlet"]`` is accurate.
            # Seeding here would read singlet=False for every well and lock
            # every singlet into the ``empty`` group.

            # State.
            self.cluster_strategy = build_cluster_strategy(self.cfg)
            self.extractor: Optional[EmbeddingExtractor] = None
            self.embeddings: Dict[str, np.ndarray] = {}
            self.well_idx_in_emb: Dict[str, np.ndarray] = {}
            self._keep_indices: Optional[np.ndarray] = None
            self._embed_done = False
            self.label_tool = None  # populated after embedding finishes

            # Signals → GUI-thread slots.
            self.progress_signal.connect(self._on_progress, Qt.QueuedConnection)
            self.status_signal.connect(self._on_status, Qt.QueuedConnection)
            self.embed_done_signal.connect(self._on_embed_done, Qt.QueuedConnection)
            self.embed_failed_signal.connect(self._on_embed_failed, Qt.QueuedConnection)

            self._build_ui()

            # Defer the slow work (find_fish + embedding) by one event-loop
            # tick. The dock widget needs to paint first, otherwise the host
            # sees a blank rectangle until find_fish + checkpoint load are
            # done. ``QTimer.singleShot(0, …)`` posts the call to the back
            # of the queue — Qt processes paint events ahead of it.
            QTimer.singleShot(0, self._start_embedding)

        # -- UI scaffolding ----------------------------------------------

        def _build_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)

            # Status panel — visible during embedding, hidden after. The
            # progress bar starts in "busy" / indeterminate mode (range
            # 0, 0) so the user sees movement immediately while we're in
            # the find_fish + model-load phase; ``_on_progress`` will
            # switch it to a determinate range once batch counts arrive.
            self.status_label = QLabel("Starting…")
            self.status_label.setStyleSheet("font-weight: bold;")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)
            layout.addWidget(self.status_label)
            layout.addWidget(self.progress_bar)

            # Dev-mode loading section: spell out the steps that run before the
            # dock is interactive, so the (sometimes multi-second, e.g. the
            # first UMAP fit's numba compile) pauses read as expected work
            # rather than a hang. Shown only when dev_mock_embeddings is set.
            self._dev_mode = bool(self.cfg.get("dev_mock_embeddings", False))
            self._dev_steps_label = None
            if self._dev_mode:
                self._dev_steps_label = QLabel(
                    "<b>Dev mode — preparing Finding Dory</b><br>"
                    "1. Compute embeddings (mock, per channel)<br>"
                    "2. Find fish + extract well crops<br>"
                    "3. Fit UMAP per channel "
                    "<i>(first fit compiles UMAP — the UI may pause here)</i><br>"
                    "4. Cluster wells (HDBSCAN)<br>"
                    "5. Render scatter + fish overlay"
                )
                self._dev_steps_label.setWordWrap(True)
                self._dev_steps_label.setStyleSheet(
                    "color: #aaa; background: rgba(255,255,255,0.04);"
                    " padding: 6px; border-radius: 4px;"
                )
                layout.addWidget(self._dev_steps_label)

            # Container that gets the LabelTool widget after embedding completes.
            self.tool_container = QWidget()
            self.tool_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._tool_layout = QVBoxLayout(self.tool_container)
            self._tool_layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.tool_container, 1)

            # Save button — always visible, only enabled after embed done.
            self.save_btn = QPushButton("Save labels and CSV")
            self.save_btn.setEnabled(False)
            self.save_btn.clicked.connect(self._on_save)
            layout.addWidget(self.save_btn)

        # -- Background embedding ----------------------------------------

        def _start_embedding(self):
            self.status_label.setText("Loading model…")

            # Resolve which wells to show (singlet filter, if enabled). Done on
            # the GUI thread because it may auto-run find_fish, which touches
            # the napari points layer.
            self._apply_singlet_filter()

            # Prefer a background pre-warm started right after stitching; only
            # fall back to computing here if none is usable.
            if self._try_adopt_prewarm():
                return

            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.future = self.executor.submit(self._embed_threaded)
            self.future.add_done_callback(self._on_future_done)

        def _ensure_classified(self):
            """Run Finding Nemo's ``find_fish`` once if it hasn't populated
            ``singlet`` yet.

            This must happen regardless of ``filter_to_singlets``: it's what
            produces the empty/singlet/multiple/deformed classification that
            the global seeding (in ``_on_embed_done``) relies on. Every well's
            ``empty`` flag defaults to True until ``find_fish`` clears it for
            detected wells — skip this and the seeding sweeps every well into
            the empty group.
            """
            try:
                feat = self.classify.points_layer.features
                already_run = (
                    "singlet" in feat.columns
                    and np.asarray(feat["singlet"], dtype=bool).any()
                )
                if already_run:
                    return
                self.status_label.setText("Finding fish (intensity threshold)…")
                # find_fish touches the napari points layer so it has to run
                # on the GUI thread. Force a repaint so the status message
                # shows before the call blocks.
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents()
                self.classify.find_fish(self.classify._points())
            except Exception as e:
                log.warning(f"auto find_fish failed: {e}; classification may be incomplete.")

        def _apply_singlet_filter(self):
            """Decide ``self._keep_indices`` (the displayed well set).

            Always runs ``find_fish`` first so the empty/singlet classification
            is correct. Then, only if ``filter_to_singlets`` is set, restricts
            the view to singlet wells; otherwise keeps all wells (``None``).
            """
            self._ensure_classified()

            if not self.cfg.get("filter_to_singlets", True):
                log.info("filter_to_singlets=false; showing all wells.")
                self._keep_indices = None
                return

            try:
                feat = self.classify.points_layer.features
                if "singlet" in feat.columns:
                    mask = np.asarray(feat["singlet"], dtype=bool)
                    if mask.any():
                        keep = np.where(mask)[0].astype(np.int64)
                        self._keep_indices = keep
                        log.info(
                            f"showing {len(keep)} of {len(self.well_ids)} wells "
                            f"(filtered by singlet from Finding Nemo)."
                        )
                    else:
                        log.info("no singlets detected; showing all wells.")
            except Exception as e:
                log.warning(f"singlet filter skipped: {e}; showing all wells.")

        def _try_adopt_prewarm(self) -> bool:
            """Adopt the Classify background pre-warm if present and compatible.

            Returns True if the dock is now waiting on the pre-warm future
            (so the caller should not start its own pass).
            """
            future = getattr(self.classify, "_dory_embed_future", None)
            if future is None:
                return False
            if getattr(self.classify, "_dory_embed_error", None) is not None:
                log.info("prewarm errored earlier; computing fresh.")
                return False
            # The pre-warm must have used the same channels + model bundle,
            # otherwise its embeddings don't correspond to what we'd compute.
            if (getattr(self.classify, "_dory_channels", None) != self.channels
                    or getattr(self.classify, "_dory_mode", None) != self.mode):
                log.info("prewarm channels/mode differ from dock; computing fresh.")
                return False

            self.future = future
            if future.done():
                self.status_label.setText("Using pre-computed embeddings…")
            else:
                self.progress_bar.setRange(0, 0)  # indeterminate while finishing
                self.status_label.setText("Finishing pre-computed embeddings…")
            future.add_done_callback(self._on_prewarm_future_done)
            return True

        def _on_prewarm_future_done(self, future):
            """Adopt prewarm results (runs on the prewarm worker thread)."""
            try:
                # The Classify pre-warm future returns
                # (extractor, embeds, idx, umaps, clusters); the umaps/clusters
                # are read separately off classify in _on_embed_done. Unpack
                # only the first three so we tolerate either tuple shape.
                result = future.result()
                extractor, embeds, idx = result[0], result[1], result[2]
            except Exception as e:
                log.warning(f"prewarm result unusable ({e!r}); computing fresh.")
                self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                self.future = self.executor.submit(self._embed_threaded)
                self.future.add_done_callback(self._on_future_done)
                return
            # The pre-warm embeds every well; restrict to the singlet keep-set
            # here so ``filter_to_singlets`` is honored on adoption too.
            if self._keep_indices is not None:
                embeds, idx = _subset_embeddings(embeds, idx, self._keep_indices)
            self.extractor = extractor
            self.embeddings = embeds
            self.well_idx_in_emb = idx
            self.embed_done_signal.emit()

        def _embed_threaded(self):
            """Build the extractor (or mock), embed all channels, return results.

            Thin wrapper: gathers the napari-side inputs (mosaics, well centers,
            crop size) off the shared `Classify`, then delegates to the Qt-free
            `compute_embeddings` so the identical pass can run from the
            post-stitch background pre-warm too.
            """
            mosaics: Dict[str, np.ndarray] = {}
            for layer in self.viewer.layers:
                if not isinstance(layer, napari.layers.Image):
                    continue
                if layer.name not in self.channels:
                    continue
                mosaics[layer.name] = np.asarray(layer.data)

            centers = self.classify._points()

            def _progress_cb(step, total):
                self.progress_signal.emit(step, total)

            return compute_embeddings(
                self.cfg,
                self.mode,
                channels=self.channels,
                mosaics=mosaics,
                well_centers=centers,
                well_crop_px=tuple(self.classify.mask.shape),
                n_total=len(centers),
                keep_indices=self._keep_indices,
                progress_cb=_progress_cb,
                status_cb=self.status_signal.emit,
            )

        def _on_future_done(self, future):
            try:
                extractor, embeds, idx = future.result()
            except Exception as e:
                log.exception("embedding pipeline failed")
                self.embed_failed_signal.emit(repr(e))
                return
            self.extractor = extractor
            self.embeddings = embeds
            self.well_idx_in_emb = idx
            self.embed_done_signal.emit()

        # -- GUI-thread slots --------------------------------------------

        def _on_progress(self, step: int, total: int):
            if total > 0 and self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(step)
            self.status_label.setText(f"Computing embeddings… {step}/{total}")

        def _on_status(self, message: str):
            self.status_label.setText(message)

        def _align_precomputed(self, by_channel):
            """Reorder a per-channel pre-warm result onto the dock's rows.

            ``by_channel[ch]`` is aligned to ``classify._dory_embed_indices[ch]``
            (all wells, in pre-warm order). This reorders each onto
            ``self.well_idx_in_emb[ch]`` (the rows actually passed to
            LabelTool, possibly a singlet subset). Channels that don't line up
            are dropped — LabelTool then fits those live, so it's a pure
            speed-up with no correctness risk.
            """
            if not by_channel:
                return None
            src_idx = getattr(self.classify, "_dory_embed_indices", None)
            if not src_idx:
                return None
            out: Dict[str, np.ndarray] = {}
            for ch, arr in by_channel.items():
                if ch not in self.well_idx_in_emb or ch not in src_idx:
                    continue
                arr = np.asarray(arr)
                src = np.asarray(src_idx[ch])
                if len(arr) != len(src):
                    continue
                pos_of = {int(w): p for p, w in enumerate(src)}
                target = self.well_idx_in_emb[ch]
                rows = []
                ok = True
                for w in target:
                    p = pos_of.get(int(w))
                    if p is None:
                        ok = False
                        break
                    rows.append(arr[p])
                if ok and len(rows) == len(target):
                    out[ch] = np.asarray(rows)
            return out or None

        def _on_embed_done(self):
            self._embed_done = True
            self.progress_bar.setVisible(False)
            self.status_label.setText("Ready.")

            # Make well crops available to LabelTool for thumbnails. Classify
            # already extracted these into self.well_extract during init.
            well_crops = getattr(self.classify, "well_extract", None)
            if well_crops is None:
                # Defensive: extract now if Classify hasn't finished its
                # background extraction yet.
                self.status_label.setText("Extracting well crops…")
                well_crops = self.classify._extract_wells(
                    self.classify._points(), img_flag=True, parallel=True,
                )

            # Seed the LabelStore with Finding Nemo's globals. Done here (not
            # in ``__init__``) because ``_start_embedding`` may have just
            # auto-run ``find_fish``; reading ``feat["singlet"]`` earlier
            # would see all-False and seed every singlet into ``empty``,
            # locking them out of cluster assignment.
            try:
                feat = self.classify.points_layer.features
                if len(self.channels) > 0:
                    seed_ch = self.channels[0]
                    if "singlet" in feat.columns:
                        singlet_arr = np.asarray(feat["singlet"], dtype=bool)
                    else:
                        singlet_arr = np.zeros(len(self.well_ids), dtype=bool)
                    for flag in ("empty", "multiple", "deformed"):
                        if flag not in feat.columns:
                            continue
                        flagged = np.asarray(feat[flag], dtype=bool)
                        flagged_wids = [
                            wid for wid, sing, on in zip(self.well_ids, singlet_arr, flagged)
                            if (not sing) and on
                        ]
                        if flagged_wids:
                            self.store.assign(self.fish_line, seed_ch, flagged_wids, flag)
                    log.info(
                        f"seeded globals: singlets={int(singlet_arr.sum())}, "
                        f"non-singlets={len(self.well_ids) - int(singlet_arr.sum())}"
                    )
            except Exception as e:
                log.warning(f"could not seed globals from points_layer.features: {e}")

            # Compute per-channel normalization bounds from the labeller
            # config's percentile params (same ones the embedding extractor
            # uses), evaluated against each channel's mosaic. This matches
            # upstream LabelTool's ``WellLoader.channel_stats`` pattern and
            # gives stable, model-aligned brightness across all crops.
            #
            # Earlier attempts used the napari layer's display contrast as
            # the source, but napari's default ``(min, max)`` is the full
            # uint16 range — that's why crops rendered very dark. The model
            # config's ``low_percentile``/``high_percentile`` (typically
            # 0.5 / 99.97 for fluor) produces the right encoded RGB.
            per_channel_contrast: Dict[str, Tuple[float, float]] = {}
            try:
                from fish_sorter.helpers.embedding.normalize import (
                    ChannelContrastConfig,
                    compute_channel_stats,
                )
                contrast_block = (
                    self.cfg.get("models", {}).get(self.mode, {}).get("contrast", {})
                )
                for layer in self.viewer.layers:
                    if not isinstance(layer, napari.layers.Image):
                        continue
                    if layer.name not in self.channels:
                        continue
                    ch = layer.name
                    cfg_entry = (
                        contrast_block.get(ch)
                        or contrast_block.get(ch.upper())
                        or contrast_block.get("_FLUOR")
                    )
                    if cfg_entry is None:
                        continue
                    try:
                        ccfg = ChannelContrastConfig.from_dict(cfg_entry)
                        lo, hi = compute_channel_stats(np.asarray(layer.data), ccfg)
                        per_channel_contrast[ch] = (float(lo), float(hi))
                        log.info(
                            f"contrast bounds for {ch}: ({lo:.1f}, {hi:.1f}) "
                            f"(low={ccfg.low_percentile}%, high={ccfg.high_percentile}%)"
                        )
                    except Exception as e:
                        log.warning(f"could not compute contrast for {ch}: {e}")
            except Exception as e:
                log.warning(f"per-channel contrast snapshot failed: {e}")

            # Re-align the pre-warm's UMAP + clusters (fit on ALL wells, in
            # ``classify._dory_embed_indices`` order) onto the rows we're
            # actually handing LabelTool (``self.well_idx_in_emb``, possibly a
            # singlet subset). If anything doesn't line up we just drop it and
            # LabelTool fits live — so this is a pure speed-up, never a
            # correctness risk.
            per_channel_umap = self._align_precomputed(
                getattr(self.classify, "_dory_umap", None)
            )
            per_channel_clusters = self._align_precomputed(
                getattr(self.classify, "_dory_clusters", None)
            )

            try:
                self.label_tool = _LabelToolFactory(
                    viewer=self.viewer,
                    prefix=self.prefix,
                    channels=self.channels,
                    well_ids=self.well_ids,
                    well_names=self.well_names,
                    well_crops=well_crops,
                    per_channel_embeddings=self.embeddings,
                    per_channel_indices=self.well_idx_in_emb,
                    cluster_strategy=self.cluster_strategy,
                    store=self.store,
                    per_channel_contrast=per_channel_contrast,
                    umap_cfg=self.cfg.get("umap", {}),
                    per_channel_umap=per_channel_umap,
                    per_channel_clusters=per_channel_clusters,
                )
            except Exception as e:
                log.exception("LabelTool construction failed")
                self._on_embed_failed(repr(e))
                return

            self.label_tool.save_requested.connect(self._on_save)
            self._tool_layout.addWidget(self.label_tool)
            self.save_btn.setEnabled(True)
            self.status_label.setVisible(False)
            if self._dev_steps_label is not None:
                self._dev_steps_label.setVisible(False)

        def _on_embed_failed(self, message: str):
            self._embed_done = False
            self.status_label.setText("Embedding failed.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(
                self,
                "Finding Dory: embedding failed",
                f"Embedding pipeline raised:\n\n{message}\n\n"
                f"Check the log for the full traceback.",
            )

        # -- Save --------------------------------------------------------

        def _on_save(self):
            path = default_csv_path(self.expt_dir, self.prefix)

            # Source of truth for empty/singlet/multiple/deformed/lHead is
            # Finding Nemo's points_layer.features. Build per-well overrides
            # so write_wide_csv layers them on top of LabelStore-derived
            # defaults.
            well_defaults: Dict[str, Dict[str, int]] = {}
            try:
                feat = self.classify.points_layer.features
                default_cols = [
                    c for c in ("empty", "singlet", "multiple", "deformed", "lHead")
                    if c in feat.columns
                ]
                for i, wid in enumerate(self.well_ids):
                    well_defaults[wid] = {
                        c: int(bool(feat[c].iloc[i])) for c in default_cols
                    }
            except Exception as e:
                log.warning(
                    f"could not read defaults from points_layer.features: {e}; "
                    f"falling back to LabelStore-derived defaults."
                )
                well_defaults = {}

            try:
                write_wide_csv(
                    store=self.store,
                    well_order=self.well_ids,
                    lhead_map={},  # lHead now comes via well_defaults
                    channels=self.channels,
                    fish_line=self.fish_line,
                    path=path,
                    well_defaults=well_defaults,
                )
            except Exception as e:
                log.exception("save failed")
                QMessageBox.critical(self, "Save failed", str(e))
                return
            self.status_label.setVisible(True)
            self.status_label.setText(f"Saved -> {path}")
            log.info(f"Finding Dory saved CSV to {path}")

        # -- Cleanup -----------------------------------------------------

        def cleanup(self):
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass
            if self.label_tool is not None:
                try:
                    self.label_tool.cleanup()
                except Exception:
                    log.exception("label_tool cleanup failed")

        def closeEvent(self, event):
            # Closing the dock fires the widget's close event before
            # ``destroyed`` — restore viewer state here so the host's mosaics
            # and "Well Locations" points become visible again.
            self.cleanup()
            super().closeEvent(event)

    return FindingDory


def FindingDory(cfg_dir: Path, classify, parent=None):
    """Factory — defers Qt/matplotlib/napari imports until first call.

    Behaves like a class at the call site: `FindingDory(cfg_dir, classify)`
    returns an instance of the inner `FindingDory(QWidget)` class.
    """
    cls = _build_finding_dory()
    return cls(cfg_dir, classify, parent=parent)
