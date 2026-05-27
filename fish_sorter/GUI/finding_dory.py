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
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.path import Path as MplPath
    from matplotlib.widgets import LassoSelector
    from qtpy.QtCore import Qt, QTimer, Signal
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QShortcut,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    from fish_sorter.helpers.embedding.clustering import build_cluster_strategy
    from fish_sorter.helpers.embedding.extractor import EmbeddingExtractor, load_config
    from fish_sorter.helpers.labelling.fish_line import parse_fish_line

    class FindingDory(QWidget):
        """Embedding-based well labelling dock.

        Construction kicks off a background thread that loads the model and
        computes per-channel embeddings, then a UMAP + cluster pass for the
        active channel. The UI is interactive only after that thread completes;
        in the meantime the status panel shows progress.
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

            # Well metadata for the store.
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
            # Ensure every (fish_line, channel) scope exists up front.
            for ch in self.channels:
                self.store._get_scope(f"{self.fish_line}|{ch}")

            # State.
            self.cluster_strategy = build_cluster_strategy(self.cfg)
            self.extractor: Optional[EmbeddingExtractor] = None
            self.embeddings: Dict[str, np.ndarray] = {}
            self.well_idx_in_emb: Dict[str, np.ndarray] = {}
            self._umap_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            self.lhead_map: Dict[str, bool] = {}
            self.current_channel = self.channels[0]
            self.current_well = 0  # index into self.well_ids
            self._lasso: Optional[LassoSelector] = None
            self._scatter = None  # matplotlib PathCollection
            self._embed_done = False

            # Signals -> GUI-thread slots.
            self.progress_signal.connect(self._on_progress, Qt.QueuedConnection)
            self.status_signal.connect(self._on_status, Qt.QueuedConnection)
            self.embed_done_signal.connect(self._on_embed_done, Qt.QueuedConnection)
            self.embed_failed_signal.connect(self._on_embed_failed, Qt.QueuedConnection)

            # Build the UI.
            self._build_ui(FigureCanvas, Figure)

            # Hook into napari's points layer to track the focused well.
            if classify.points_layer is not None:
                classify.points_layer.events.highlight.connect(self._on_napari_select)

            # Auto-run find_orientation (no-op if no singlets are flagged yet —
            # find_orientation only operates on wells where features['singlet']
            # is True; an empty result keeps lhead_map at defaults).
            try:
                self.classify.find_orientation()
            except Exception as e:
                log.warning(f"find_orientation skipped at startup: {e}")
            self._refresh_lhead_from_classify()

            # Kick off background embedding.
            self._start_embedding()

        # -- UI construction ---------------------------------------------

        def _build_ui(self, FigureCanvas, Figure):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)

            # 1. Status panel
            self.status_label = QLabel("Starting…")
            self.status_label.setStyleSheet("font-weight: bold;")
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            layout.addWidget(self.status_label)
            layout.addWidget(self.progress_bar)

            # (Pre-embed find_fish + per-well lHead toggle live in Finding Nemo.
            # Finding Dory consumes their results via points_layer.features at
            # _start_embedding and _on_save time; it never sets them itself.)

            # 2. Channel selector
            ch_row = QHBoxLayout()
            ch_row.addWidget(QLabel("Channel:"))
            self.channel_combo = QComboBox()
            self.channel_combo.addItems(self.channels)
            self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
            ch_row.addWidget(self.channel_combo, 1)
            layout.addLayout(ch_row)

            # 2b. Color mode — swap scatter coloring between group assignment
            #     and the HDBSCAN cluster labels produced when the embedding
            #     ran.
            color_row = QHBoxLayout()
            color_row.addWidget(QLabel("Color by:"))
            self.color_mode_combo = QComboBox()
            self.color_mode_combo.addItems(["Group assignment", "HDBSCAN cluster"])
            self.color_mode_combo.currentTextChanged.connect(lambda *_: self._render_scatter())
            color_row.addWidget(self.color_mode_combo, 1)
            layout.addLayout(color_row)

            # 3. UMAP scatter (matplotlib canvas, hidden until embedding done)
            self.figure = Figure(figsize=(4, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("Embeddings not ready yet…")
            layout.addWidget(self.canvas, 1)

            # 4. Group manager
            self.group_list = QListWidget()
            self.group_list.setMaximumHeight(120)
            layout.addWidget(self.group_list)
            grp_row = QHBoxLayout()
            self.add_group_btn = QPushButton("+ Add")
            self.add_group_btn.clicked.connect(self._on_add_group)
            self.rename_group_btn = QPushButton("Rename")
            self.rename_group_btn.clicked.connect(self._on_rename_group)
            self.delete_group_btn = QPushButton("Delete")
            self.delete_group_btn.clicked.connect(self._on_delete_group)
            grp_row.addWidget(self.add_group_btn)
            grp_row.addWidget(self.rename_group_btn)
            grp_row.addWidget(self.delete_group_btn)
            layout.addLayout(grp_row)

            # 5. Lasso button
            self.lasso_btn = QPushButton("Lasso -> assign to selected group")
            self.lasso_btn.setCheckable(True)
            self.lasso_btn.toggled.connect(self._on_lasso_toggle)
            self.lasso_btn.setEnabled(False)
            layout.addWidget(self.lasso_btn)

            # 5b. Cluster picker — pick any HDBSCAN cluster and assign every
            #     well in it to the currently-selected group in one click.
            #     Populated automatically when embedding completes.
            cluster_row = QHBoxLayout()
            cluster_row.addWidget(QLabel("Cluster:"))
            self.cluster_combo = QComboBox()
            cluster_row.addWidget(self.cluster_combo, 1)
            self.cluster_assign_btn = QPushButton("-> assign to selected group")
            self.cluster_assign_btn.clicked.connect(self._on_assign_cluster)
            self.cluster_assign_btn.setEnabled(False)
            cluster_row.addWidget(self.cluster_assign_btn)
            layout.addLayout(cluster_row)

            # 6. Current-well state strip
            strip_row = QHBoxLayout()
            self.state_chips = {}
            for name in ("empty", "singlet", "multiple", "deformed", "lHead"):
                chip = QLabel(name)
                chip.setStyleSheet(
                    "border: 1px solid #888; border-radius: 3px; "
                    "padding: 2px 6px; color: #888;"
                )
                chip.setAlignment(Qt.AlignCenter)
                self.state_chips[name] = chip
                strip_row.addWidget(chip)
            layout.addLayout(strip_row)

            # 7. Save button
            self.save_btn = QPushButton("Save labels and CSV")
            self.save_btn.clicked.connect(self._on_save)
            layout.addWidget(self.save_btn)

            self._refresh_group_list()
            self._refresh_state_strip()

            # Keyboard shortcuts: L (lasso), Esc (cancel lasso), Ctrl+S (save).
            # lHead is set in Finding Nemo, not here, so no H shortcut.
            QShortcut(QKeySequence("L"), self).activated.connect(self._kb_lasso)
            QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self._kb_cancel_lasso)
            QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._on_save)

        # -- Keyboard shortcut handlers ----------------------------------

        def _kb_lasso(self):
            if self.lasso_btn.isEnabled():
                self.lasso_btn.toggle()

        def _kb_cancel_lasso(self):
            if self.lasso_btn.isChecked():
                self.lasso_btn.setChecked(False)

        # -- Background embedding ----------------------------------------

        def _start_embedding(self):
            self.status_signal.emit("Loading model…")

            self._keep_indices: Optional[np.ndarray] = None
            try:
                feat = self.classify.points_layer.features
                # If Finding Nemo hasn't already populated singlet, run
                # find_fish here ourselves — it's just intensity thresholding,
                # not the model, so it's fast (~1s) and avoids the manual
                # button click.
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
                    feat = self.classify.points_layer.features  # refresh after find_fish

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

            # lHead values also come from Finding Nemo's pass (via find_orientation
            # chained off find_fish). Refresh here so Save can include them.
            self._refresh_lhead_from_classify()

            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self.future = self.executor.submit(self._embed_threaded)
            self.future.add_done_callback(self._on_future_done)

        def _embed_threaded(self):
            """Heavy work: build extractor, embed every channel, UMAP for current.

            Honors `cfg["dev_mock_embeddings"]=true`, which skips the model
            forward pass entirely and emits random vectors of the model's
            output dim. Useful for iterating on the dock UI without waiting
            tens of minutes per click. Embeddings are meaningless under mock,
            so clusters and assignments are toy-quality — never enable in
            production.
            """
            mock = bool(self.cfg.get("dev_mock_embeddings", False))

            if mock:
                self.status_signal.emit("Mock embeddings (dev mode)…")
                n_total = len(self.classify._points())
                keep = self._keep_indices if self._keep_indices is not None else np.arange(n_total)
                # Use the configured output dim (FishDINOv3 with gem pooling = 2 * embed_dim).
                model_cfg = self.cfg["models"][self.mode]
                emb_dim = 2 * int(model_cfg.get("embedding_dim", 384))
                rng = np.random.default_rng(0)
                embeds = {ch: rng.standard_normal((len(keep), emb_dim)).astype(np.float32)
                          for ch in self.channels}
                idx = {ch: np.asarray(keep, dtype=np.int64) for ch in self.channels}
                extractor = None
            else:
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
            # UMAP for the currently selected channel only — others lazy.
            xy, cluster_labels = self._compute_umap(embeds[self.current_channel])

            return extractor, embeds, idx, xy, cluster_labels

        def _on_future_done(self, future):
            """Worker thread finished — hop to GUI thread via QTimer."""
            try:
                extractor, embeds, idx, xy, cluster_labels = future.result()
            except Exception as e:
                log.exception("embedding pipeline failed")
                self.embed_failed_signal.emit(repr(e))
                return

            # Store results before signalling the GUI.
            self.extractor = extractor
            self.embeddings = embeds
            self.well_idx_in_emb = idx
            self._umap_cache[self.current_channel] = (xy, cluster_labels)
            self.embed_done_signal.emit()

        def _compute_umap(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Per-channel UMAP + clustering. Runs on worker thread."""
            from umap import UMAP
            umap_cfg = dict(self.cfg.get("umap", {}))
            reducer = UMAP(n_components=2, **umap_cfg)
            xy = reducer.fit_transform(emb)
            labels = self.cluster_strategy.cluster(emb)
            return xy.astype(np.float32), labels

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
            self.lasso_btn.setEnabled(True)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.status_label.setText(
                "Ready — pick a group then drag a lasso on the plot below."
            )
            self._render_scatter()

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

        # -- Channel switching -------------------------------------------

        def _on_channel_changed(self, channel: str):
            if channel not in self.channels:
                return
            self.current_channel = channel
            if not self._embed_done:
                return
            if channel not in self._umap_cache:
                # Compute on demand. Runs on GUI thread — small per-channel cost
                # (UMAP on ~600 × 1536 takes ~1-3s); acceptable for now.
                self.status_label.setText(f"Computing UMAP for {channel}…")
                QTimer.singleShot(0, lambda: self._compute_and_render_channel(channel))
            else:
                self._render_scatter()
            self._refresh_group_list()

        def _compute_and_render_channel(self, channel: str):
            try:
                xy, labels = self._compute_umap(self.embeddings[channel])
            except Exception as e:
                log.exception("UMAP on channel switch failed")
                QMessageBox.warning(self, "UMAP failed", str(e))
                return
            self._umap_cache[channel] = (xy, labels)
            self.status_label.setText("Ready.")
            self._render_scatter()

        # -- Scatter rendering + lasso -----------------------------------

        def _render_scatter(self):
            ch = self.current_channel
            if ch not in self._umap_cache:
                return
            xy, cluster_labels = self._umap_cache[ch]
            wids = self._well_ids_for_channel(ch)

            mode = self.color_mode_combo.currentText()
            if mode == "HDBSCAN cluster":
                colors = self._colors_for_clusters(cluster_labels)
                title = f"UMAP - {ch} (HDBSCAN clusters)"
            else:
                colors = self._colors_for_wids(wids, ch)
                title = f"UMAP - {ch} (group assignments)"

            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(title)
            self._scatter = self.ax.scatter(
                xy[:, 0], xy[:, 1], c=colors, s=14, edgecolors="none"
            )
            self.canvas.draw_idle()
            self._refresh_cluster_combo(cluster_labels)

            # Re-attach lasso if it was enabled.
            if self.lasso_btn.isChecked():
                self._enable_lasso()

        def _colors_for_clusters(self, cluster_labels: np.ndarray) -> np.ndarray:
            """Map each well to a tab20 color by its HDBSCAN cluster label.

            `-1` is noise (no cluster) and renders as muted grey.
            """
            import matplotlib.cm as cm
            palette = cm.get_cmap("tab20")
            n = len(cluster_labels)
            out = np.zeros((n, 4), dtype=np.float32)
            for i, label in enumerate(cluster_labels):
                lbl = int(label)
                if lbl < 0:
                    out[i] = [0.60, 0.60, 0.60, 0.6]
                else:
                    out[i] = palette(lbl % 20)
            return out

        def _refresh_cluster_combo(self, cluster_labels: np.ndarray):
            """Re-populate the cluster dropdown with one entry per HDBSCAN cluster."""
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            unique = sorted(set(int(l) for l in cluster_labels))
            for lbl in unique:
                count = int((cluster_labels == lbl).sum())
                name = "noise" if lbl == -1 else f"cluster {lbl}"
                self.cluster_combo.addItem(f"{name} ({count} wells)", userData=lbl)
            self.cluster_combo.blockSignals(False)
            self.cluster_assign_btn.setEnabled(self._embed_done and self.cluster_combo.count() > 0)

        def _on_assign_cluster(self):
            """Assign every well in the picked HDBSCAN cluster to the selected group."""
            group = self._currently_selected_group()
            if group is None:
                QMessageBox.information(
                    self, "No group selected", "Pick a group from the list above first."
                )
                return
            idx = self.cluster_combo.currentIndex()
            if idx < 0:
                return
            cluster_id = self.cluster_combo.itemData(idx)
            ch = self.current_channel
            if ch not in self._umap_cache:
                return
            _, labels = self._umap_cache[ch]
            wids_in_order = self._well_ids_for_channel(ch)
            selected = [
                wids_in_order[i] for i, l in enumerate(labels) if int(l) == int(cluster_id)
            ]
            if not selected:
                return
            self.store.assign(self.fish_line, ch, selected, group)
            if group in GLOBAL_GROUPS:
                self._sync_globals_to_features()
            label_name = "noise" if int(cluster_id) == -1 else f"cluster {cluster_id}"
            self.status_label.setText(
                f"Assigned {label_name} ({len(selected)} wells) to '{group}' in {ch}."
            )
            self._render_scatter()
            self._refresh_state_strip()

        def _well_ids_for_channel(self, ch: str) -> List[str]:
            idx = self.well_idx_in_emb.get(ch)
            if idx is None:
                return self.well_ids
            return [self.well_ids[int(i)] for i in idx]

        def _colors_for_wids(self, wids: List[str], ch: str) -> np.ndarray:
            asgn = self.store.assignments(self.fish_line, ch)
            colors = np.zeros((len(wids), 4), dtype=np.float32)
            for i, wid in enumerate(wids):
                group = asgn.get(wid)
                if group is None:
                    colors[i] = [0.60, 0.60, 0.60, 0.6]  # unassigned grey
                else:
                    colors[i] = self.store.group_color(self.fish_line, ch, group)
            return colors

        def _on_lasso_toggle(self, on: bool):
            if on:
                self._enable_lasso()
            else:
                self._disable_lasso()

        def _enable_lasso(self):
            if self._scatter is None:
                return
            self._disable_lasso()
            self._lasso = LassoSelector(self.ax, onselect=self._on_lasso_select)
            self.canvas.draw_idle()

        def _disable_lasso(self):
            if self._lasso is not None:
                self._lasso.disconnect_events()
                self._lasso = None

        def _on_lasso_select(self, verts):
            if self._scatter is None:
                return
            mpl_path = MplPath(verts)
            xy = self._scatter.get_offsets()
            mask = mpl_path.contains_points(xy)
            ch = self.current_channel
            wids_in_order = self._well_ids_for_channel(ch)
            selected = [wids_in_order[i] for i, m in enumerate(mask) if m]

            if not selected:
                self.status_label.setText("Lasso was empty — try again.")
                return
            if len(selected) > 500:
                ans = QMessageBox.question(
                    self,
                    "Confirm large assignment",
                    f"Assign {len(selected)} wells to the selected group?",
                )
                if ans != QMessageBox.Yes:
                    return

            group = self._currently_selected_group()
            if group is None:
                QMessageBox.information(
                    self,
                    "No group selected",
                    "Pick a group from the list above before lassoing.",
                )
                return
            self.store.assign(self.fish_line, ch, selected, group)
            if group in GLOBAL_GROUPS:
                # Globals (empty/multiple/deformed) need to round-trip back to
                # Finding Nemo's shared features so its dock shows the override.
                self._sync_globals_to_features()
            self.status_label.setText(
                f"Assigned {len(selected)} wells to '{group}' in {ch}."
            )
            self._render_scatter()
            self._refresh_state_strip()

        def _sync_globals_to_features(self):
            """Mirror LabelStore global-group assignments into points_layer.features.

            Finding Dory can override Finding Nemo's empty/multiple/deformed
            via lasso. This pushes the LabelStore truth back into the shared
            features DataFrame so the Finding Nemo dock reflects it immediately.
            Wells assigned to a global are also marked singlet=False; wells the
            user later unassigns (via unassign action — not implemented yet) are
            left alone and need a separate sweep.
            """
            try:
                feat = self.classify.points_layer.features
            except AttributeError:
                return
            wid_to_idx = {wid: i for i, wid in enumerate(self.well_ids)}
            for ch in self.channels:
                for wid, group in self.store.assignments(self.fish_line, ch).items():
                    if group not in GLOBAL_GROUPS:
                        continue
                    idx = wid_to_idx.get(wid)
                    if idx is None:
                        continue
                    for g in ("empty", "multiple", "deformed"):
                        if g in feat.columns:
                            feat.loc[idx, g] = (group == g)
                    if "singlet" in feat.columns:
                        feat.loc[idx, "singlet"] = False

        # -- Group manager -----------------------------------------------

        def _refresh_group_list(self):
            """Populate the group list, with global groups shown bold at the top.

            Globals (`empty`, `multiple`, `deformed`) come pre-populated from
            Finding Nemo, but Finding Dory can override them via lasso → assign.
            They're shown bold + non-renamable but selectable.
            """
            self.group_list.clear()
            ch = self.current_channel
            for g in self.store.groups(self.fish_line, ch):
                item = QListWidgetItem(g)
                if g in GLOBAL_GROUPS:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.group_list.addItem(item)

        def _currently_selected_group(self) -> Optional[str]:
            row = self.group_list.currentRow()
            if row < 0:
                return None
            item = self.group_list.item(row)
            return item.text() if item else None

        def _on_add_group(self):
            name, ok = QInputDialog.getText(self, "New group", "Name this phenotype:")
            name = name.strip()
            if not ok or not name:
                return
            if not self.store.create_group(self.fish_line, self.current_channel, name):
                QMessageBox.warning(self, "Group exists", f"'{name}' already exists.")
                return
            self._refresh_group_list()

        def _on_rename_group(self):
            old = self._currently_selected_group()
            if old is None or old in GLOBAL_GROUPS:
                QMessageBox.warning(
                    self, "Can't rename", "Default groups can't be renamed."
                )
                return
            new, ok = QInputDialog.getText(self, "Rename group", "New name:", text=old)
            new = new.strip()
            if not ok or not new or new == old:
                return
            if not self.store.rename_group(self.fish_line, self.current_channel, old, new):
                QMessageBox.warning(self, "Rename failed", "Name conflict or unknown group.")
                return
            self._refresh_group_list()
            self._render_scatter()

        def _on_delete_group(self):
            grp = self._currently_selected_group()
            if grp is None or grp in GLOBAL_GROUPS:
                QMessageBox.warning(
                    self, "Can't delete", "Default groups can't be deleted."
                )
                return
            ans = QMessageBox.question(
                self, "Delete group", f"Delete '{grp}' and unassign its wells?"
            )
            if ans != QMessageBox.Yes:
                return
            n = self.store.delete_group(self.fish_line, self.current_channel, grp)
            self.status_label.setText(f"Deleted '{grp}' ({n} wells unassigned).")
            self._refresh_group_list()
            self._render_scatter()

        # -- Per-well state ----------------------------------------------

        def _on_napari_select(self, _event):
            if self.classify.points_layer is None:
                return
            selected = self.classify.points_layer.selected_data
            if not selected:
                return
            self.current_well = int(next(iter(selected)))
            self._refresh_state_strip()

        def _refresh_lhead_from_classify(self):
            """Copy lHead values out of Finding Nemo's shared points_layer.features.

            Kept for backward compat — `lhead_map` is also rebuilt at save
            time from `points_layer.features`, so call sites that go through
            `_on_save` don't strictly need it.
            """
            try:
                feat = self.classify.points_layer.features
            except AttributeError:
                return
            if "lHead" not in feat.columns:
                return
            for i, wid in enumerate(self.well_ids):
                self.lhead_map[wid] = bool(feat["lHead"].iloc[i])

        def _refresh_state_strip(self):
            """Update the read-only chip strip from Finding Nemo's features.

            `empty`/`singlet`/`multiple`/`deformed`/`lHead` are owned by
            Finding Nemo; we display them here but never write to them.
            """
            if not (0 <= self.current_well < len(self.well_ids)):
                return
            try:
                feat = self.classify.points_layer.features
            except AttributeError:
                return
            for name in ("empty", "singlet", "multiple", "deformed", "lHead"):
                if name in feat.columns:
                    val = int(bool(feat[name].iloc[self.current_well]))
                else:
                    val = 0
                chip = self.state_chips[name]
                if val:
                    chip.setStyleSheet(
                        "border: 1px solid #2a9d2a; border-radius: 3px; "
                        "padding: 2px 6px; color: white; background: #2a9d2a;"
                    )
                else:
                    chip.setStyleSheet(
                        "border: 1px solid #888; border-radius: 3px; "
                        "padding: 2px 6px; color: #888;"
                    )

        # -- Save --------------------------------------------------------

        def _on_save(self):
            path = default_csv_path(self.expt_dir, self.prefix)

            # Source of truth for empty/singlet/multiple/deformed/lHead is
            # Finding Nemo's points_layer.features. Build a per-well dict so
            # write_wide_csv overlays those values onto its output.
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
                    lhead_map=self.lhead_map,
                    channels=self.channels,
                    fish_line=self.fish_line,
                    path=path,
                    well_defaults=well_defaults,
                )
            except Exception as e:
                log.exception("save failed")
                QMessageBox.critical(self, "Save failed", str(e))
                return
            self.status_label.setText(f"Saved -> {path}")
            log.info(f"Finding Dory saved CSV to {path}")

        # -- Cleanup -----------------------------------------------------

        def cleanup(self):
            self._disable_lasso()
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass

    return FindingDory


def FindingDory(cfg_dir: Path, classify, parent=None):
    """Factory — defers Qt/matplotlib/napari imports until first call.

    Behaves like a class at the call site: `FindingDory(cfg_dir, classify)`
    returns an instance of the inner `FindingDory(QWidget)` class.
    """
    cls = _build_finding_dory()
    return cls(cfg_dir, classify, parent=parent)
