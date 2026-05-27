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
    from fish_sorter.helpers.embedding.extractor import EmbeddingExtractor, load_config
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
            mode_map = self.cfg.get("pick_type_to_mode", {})
            self.mode = mode_map.get(self.pick_type, "fish")
            if self.pick_type not in mode_map:
                log.warning(
                    f"pick_type {self.pick_type!r} not in pick_type_to_mode; "
                    f"falling back to mode={self.mode!r}"
                )

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
            self._start_embedding()

        # -- UI scaffolding ----------------------------------------------

        def _build_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)

            # Status panel — visible during embedding, hidden after.
            self.status_label = QLabel("Starting…")
            self.status_label.setStyleSheet("font-weight: bold;")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            layout.addWidget(self.status_label)
            layout.addWidget(self.progress_bar)

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
            self.status_signal.emit("Loading model…")

            # Auto-run find_fish if Finding Nemo hasn't populated singlet yet.
            try:
                feat = self.classify.points_layer.features
                already_run = (
                    "singlet" in feat.columns
                    and np.asarray(feat["singlet"], dtype=bool).any()
                )
                if not already_run:
                    self.status_label.setText("Finding fish (intensity threshold)…")
                    try:
                        self.classify.find_fish(self.classify._points())
                    except Exception as e:
                        log.warning(f"auto find_fish failed: {e}; embedding all wells.")
                    feat = self.classify.points_layer.features

                if "singlet" in feat.columns:
                    mask = np.asarray(feat["singlet"], dtype=bool)
                    if mask.any():
                        keep = np.where(mask)[0].astype(np.int64)
                        self._keep_indices = keep
                        log.info(
                            f"embedding {len(keep)} of {len(self.well_ids)} wells "
                            f"(filtered by singlet from Finding Nemo)."
                        )
                    else:
                        log.info("no singlets detected; embedding all wells.")
            except Exception as e:
                log.warning(f"singlet pre-filter skipped: {e}; embedding all wells.")

            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.future = self.executor.submit(self._embed_threaded)
            self.future.add_done_callback(self._on_future_done)

        def _embed_threaded(self):
            """Build the extractor (or mock), embed all channels, return results.

            Honors `cfg["dev_mock_embeddings"]=true` — generates well-separated
            synthetic clusters per channel, skipping the model entirely. Lets
            you iterate on the dock UI in seconds instead of tens of minutes.
            """
            mock = bool(self.cfg.get("dev_mock_embeddings", False))

            if mock:
                self.status_signal.emit("Mock embeddings (dev mode)…")
                n_total = len(self.classify._points())
                keep = self._keep_indices if self._keep_indices is not None else np.arange(n_total)
                model_cfg = self.cfg["models"][self.mode]
                emb_dim = 2 * int(model_cfg.get("embedding_dim", 384))
                n_clusters = 6
                n_wells = len(keep)
                embeds: Dict[str, np.ndarray] = {}
                idx: Dict[str, np.ndarray] = {}
                for ch_idx, ch in enumerate(self.channels):
                    rng = np.random.default_rng(ch_idx + 1)
                    centers = rng.standard_normal((n_clusters, emb_dim)).astype(np.float32) * 15.0
                    assignments = rng.integers(0, n_clusters, size=n_wells)
                    noise = rng.standard_normal((n_wells, emb_dim)).astype(np.float32) * 0.3
                    embeds[ch] = centers[assignments] + noise
                    idx[ch] = np.asarray(keep, dtype=np.int64)
                return None, embeds, idx

            self.status_signal.emit("Loading checkpoint…")
            extractor = EmbeddingExtractor(self.cfg, mode=self.mode)
            self.status_signal.emit("Computing embeddings…")

            mosaics: Dict[str, np.ndarray] = {}
            for layer in self.viewer.layers:
                if not isinstance(layer, napari.layers.Image):
                    continue
                if layer.name not in self.channels:
                    continue
                mosaics[layer.name] = np.asarray(layer.data)

            centers = self.classify._points()
            well_crop_px = tuple(self.classify.mask.shape)

            def _progress_cb(step, total):
                self.progress_signal.emit(step, total)

            embeds, idx = extractor.extract_from_mosaic(
                mosaics=mosaics,
                well_centers_px=centers,
                well_crop_px=well_crop_px,
                well_indices_to_embed=self._keep_indices,
                progress_cb=_progress_cb,
            )
            return extractor, embeds, idx

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
                )
            except Exception as e:
                log.exception("LabelTool construction failed")
                self._on_embed_failed(repr(e))
                return

            self.label_tool.save_requested.connect(self._on_save)
            self._tool_layout.addWidget(self.label_tool)
            self.save_btn.setEnabled(True)
            self.status_label.setVisible(False)

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
