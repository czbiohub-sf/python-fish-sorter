"""Embeddable napari labeller dock — vendored from zebra ``LabelTool``.

Vendored from
``zebrafish-unsupervised-classification/fish_classify/labelling/label_tool.py``
lines 289-end (the ``LabelTool`` class). The ``LabelStore`` data model
(upstream lines 1-281) is vendored separately in
``fish_sorter.helpers.labelling.store`` — import from there, do NOT re-vendor.

Eight refactor items (see PR description) applied surgically:

1. The ``QApplication`` instantiation block in upstream ``__init__`` is dropped.
   ``FindingDory`` constructs us with a parent already alive in a ``QApplication``.
2. ``LabelTool.run()`` is replaced with ``as_dock_widget()`` returning ``self``.
   The parent owns the event loop.
3. ``napari.Viewer`` is injected by the caller, never created here.
4. ``well_loaders: Dict[str, object]`` (per-experiment loaders) is replaced
   with ``well_crops: List[Dict[channel, np.ndarray]]`` — preloaded by
   ``FindingDory`` from the napari Image layers.
5. ``mode``/``_fluor_cols`` derivation is dropped; channels come in via the
   constructor.
6. Single-experiment simplification: one ``fish_line`` synthesized from
   ``prefix``; the multi-line tab/combo UI is reduced to a static label.
7. Direct ``hdbscan.HDBSCAN``/``EmbeddingClusterer`` use is replaced with
   the injected ``ClusterStrategy``.
8. ``_on_save`` / ``_on_load`` are dropped; ``FindingDory`` handles persistence
   via ``write_wide_csv`` and connects to ``save_requested``.

Subsystems dropped wholesale (out of scope for fish-sorter):

- Fine-tune worker (``fish_classify.clustering.fine_tune_worker``).
- Match panel / learn panel side docks.
- Cross-channel grid mode (``_enter_cross_channel`` and all helpers).
- Multi-experiment fish-line tab UI (replaced with a read-only label).
- ``clusters_as_classes`` constructor parameter.
- ``LabelStore.save_csv`` / ``load_csv`` UI wiring.
- Composite-crop preloader and Fish-UMAP contrast slider (the upstream
  Fish-UMAP overlay system is kept, but uses the active-channel cached
  crop from ``self._well_crops`` so the user still sees fish thumbnails
  laid out on the UMAP).

The destination is self-contained: no imports from ``fish_classify.*`` allowed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fish_sorter.helpers.embedding.clustering import ClusterStrategy  # noqa: F401 (type)
from fish_sorter.helpers.labelling.fish_line import parse_fish_line
from fish_sorter.helpers.labelling.store import GLOBAL_GROUPS, LabelStore, _scope_key  # noqa: F401

log = logging.getLogger(__name__)

DEFAULT_GROUPS = ["empty", "multiple", "deformed"]

# Distinct colours for groups (tab20 palette, RGBA float).
_TAB20 = [
    [0.12, 0.47, 0.71, 1.0],
    [1.00, 0.50, 0.05, 1.0],
    [0.17, 0.63, 0.17, 1.0],
    [0.84, 0.15, 0.16, 1.0],
    [0.58, 0.40, 0.74, 1.0],
    [0.55, 0.34, 0.29, 1.0],
    [0.89, 0.47, 0.76, 1.0],
    [0.50, 0.50, 0.50, 1.0],
    [0.74, 0.74, 0.13, 1.0],
    [0.09, 0.75, 0.81, 1.0],
    [0.68, 0.78, 0.91, 1.0],
    [1.00, 0.73, 0.47, 1.0],
    [0.60, 0.87, 0.54, 1.0],
    [1.00, 0.60, 0.59, 1.0],
    [0.77, 0.69, 0.84, 1.0],
    [0.77, 0.61, 0.58, 1.0],
    [0.97, 0.71, 0.85, 1.0],
    [0.78, 0.78, 0.78, 1.0],
    [0.86, 0.86, 0.55, 1.0],
    [0.62, 0.85, 0.90, 1.0],
]

_NOISE_COLOR = [0.35, 0.35, 0.35, 0.5]
_UNASSIGNED_COLOR = [0.60, 0.60, 0.60, 0.6]


def _uint16_to_rgb(
    crop: np.ndarray,
    rgb_color: Tuple[float, float, float],
    low: Optional[float] = None,
    high: Optional[float] = None,
    dark_on_white: bool = False,
) -> np.ndarray:
    """Normalize a uint16 2-D crop and paint it with the channel's RGB color.

    Mirrors the rendering ``label_tool.py`` upstream does inside
    ``_preload_crops``/``_show_crop`` — but on a single cached crop, not a
    full mosaic. Percentile bounds default to 1/99 of the crop itself
    when not supplied, which is fine for thumbnails.
    """
    arr = np.asarray(crop)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D crop, got shape {arr.shape}")
    if low is None or high is None:
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size:
            low_v, high_v = np.percentile(finite, [1.0, 99.0])
        else:
            low_v, high_v = 0.0, 65535.0
        low = float(low_v if low is None else low)
        high = float(high_v if high is None else high)
    rng = high - low if high > low else 1.0
    normalized = np.clip((arr.astype(np.float32) - low) / rng, 0.0, 1.0)
    h, w = normalized.shape
    r_w, g_w, b_w = rgb_color
    if dark_on_white:
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        for c, weight in enumerate([r_w, g_w, b_w]):
            if weight > 0:
                rgb[:, :, c] = (255 - normalized * weight * 255).astype(np.uint8)
            else:
                rgb[:, :, c] = (255 - normalized * 255).astype(np.uint8)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c, weight in enumerate([r_w, g_w, b_w]):
            if weight > 0:
                rgb[:, :, c] = (normalized * weight * 255).astype(np.uint8)
    return rgb


def _build_label_tool():
    """Defer heavy imports (qtpy/napari/matplotlib/PIL) until first call.

    Matches the deferred-import pattern in
    ``fish_sorter.GUI.finding_dory._build_finding_dory``.
    """
    from qtpy.QtCore import Qt, QSize, Signal
    from qtpy.QtGui import QImage, QPixmap
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    from fish_sorter.helpers.embedding.channel_mapping import get_channel_display

    class _ShrinkableLabel(QLabel):
        """QLabel that lets its parent scrollable shrink below the pixmap size."""

        def minimumSizeHint(self):
            return QSize(0, 0)

    class LabelTool(QWidget):
        """Embeddable labeller dock — refactored from zebra ``LabelTool``.

        Constructed by ``FindingDory`` after embeddings finish. All heavy
        work (model loading, embedding, UMAP, clustering) happens before
        construction; this widget only renders and handles user interaction.
        """

        save_requested = Signal()

        def __init__(
            self,
            viewer,
            prefix: str,
            channels: List[str],
            well_ids: List[str],
            well_names: List[str],
            well_crops: List[Dict[str, np.ndarray]],
            per_channel_embeddings: Dict[str, np.ndarray],
            per_channel_indices: Dict[str, np.ndarray],
            cluster_strategy,
            store: LabelStore,
            per_channel_contrast: Optional[Dict[str, Tuple[float, float]]] = None,
            parent=None,
        ):
            super().__init__(parent)

            # Refactor item 3: viewer is injected.
            self.viewer = viewer

            # Refactor item 6: synthesize one fish_line from prefix.
            self._prefix = prefix
            self._fish_line = parse_fish_line(prefix) or "unknown"
            # Tab-like multi-experiment UI is dropped — we expose a single
            # read-only label showing this synthesized fish_line.
            self._fish_lines: List[str] = [self._fish_line]
            self._current_line: str = self._fish_line

            # Refactor item 5: channels come in via the constructor; no
            # mode-based egg/fish hardcoding.
            self._all_channels = list(channels)
            self._current_channel: str = channels[0] if channels else ""

            # Refactor item 4: per-well crops instead of per-experiment loaders.
            # well_crops[well_idx][channel] -> 2-D uint16 array.
            self._well_crops: List[Dict[str, np.ndarray]] = list(well_crops)
            self._well_ids: List[str] = list(well_ids)
            self._well_names: List[str] = list(well_names)
            # well_id -> well index (for fast lookups in callbacks).
            self._wid_to_widx: Dict[str, int] = {wid: i for i, wid in enumerate(self._well_ids)}

            # Refactor item 7: clustering goes through the injected strategy.
            self._cluster_strategy = cluster_strategy

            # Metadata DataFrame (one row per well — matches the LabelStore convention).
            self.metadata = pd.DataFrame({
                "well_id": self._well_ids,
                "well_name": self._well_names,
                "experiment": [self._prefix] * len(self._well_ids),
                "fish_line": [self._fish_line] * len(self._well_ids),
            })

            # Store is created by FindingDory and passed in.
            self.store = store
            # Register every channel for this single fish line so global
            # group propagation works (upstream behaviour).
            self.store._line_channels[self._fish_line] = list(self._all_channels)
            # Ensure scopes exist for each channel.
            for ch in self._all_channels:
                self.store._get_scope(_scope_key(self._fish_line, ch))

            # Embeddings: channel -> (N_filtered, D). per_channel_indices
            # gives the well-row index into self._well_ids that each
            # embedding row corresponds to (matches upstream convention).
            self.per_ch_emb: Dict[str, np.ndarray] = dict(per_channel_embeddings)
            self.per_ch_idx: Dict[str, np.ndarray] = dict(per_channel_indices)

            # Per-channel display contrast bounds for the Show-Fish thumbnails.
            # Upstream computes these once globally via ``WellLoader.channel_stats``;
            # FindingDory hands them in from each napari Image layer's
            # ``contrast_limits`` so all thumbnails normalize against the same
            # range as the in-viewer mosaic. Falling back to per-crop
            # percentiles inside ``_uint16_to_rgb`` makes every well look
            # uniformly bright (its own 99th percentile becomes the ceiling).
            self._channel_contrast: Dict[str, Tuple[float, float]] = dict(
                per_channel_contrast or {}
            )

            # First-clustering tracking — auto-assign cluster_N to wells only
            # the first time a (line, channel) scope is computed, so manual
            # edits and reclusters never wipe assignments. See _recompute_view.
            self._auto_assigned: set = set()

            # Re-propagate globals across channels (upstream behaviour).
            self.store._propagate_global_groups()

            # Current view state.
            self._line_mask: Optional[np.ndarray] = None
            self._view_embeddings: Optional[np.ndarray] = None
            self._view_umap: Optional[np.ndarray] = None
            self._view_clusters: Optional[np.ndarray] = None
            self._view_indices: Optional[np.ndarray] = None
            self._view_valid_mask: Optional[np.ndarray] = None

            # Selection state — wells are captured here without immediate
            # assignment; explicit Assign / Unassign buttons commit.
            self._selected_indices: set = set()
            self._selected_well_list: List[Tuple[str, str]] = []
            self._current_well_view_idx: int = 0
            self._focused_group: Optional[str] = None

            # Crop cache: (well_name, experiment) -> RGB uint8 array,
            # built from self._well_crops on view change.
            self._crop_cache: Dict[Tuple[str, str], np.ndarray] = {}

            # Lasso state.
            self._lasso_mode = False
            self._ignore_lasso_event = False

            # UI state.
            self._color_by = "group"
            self._hidden_indices: set = set()

            # Point spread state (crop-size slider).
            self._umap_ref_points_data: Optional[np.ndarray] = None
            self._umap_world_centroid: Optional[Tuple[float, float]] = None

            # Image UMAP state (thumbnails-on-scatter).
            self._umap_channel_layers: List = []
            self._umap_thumb_cache: Optional[List] = None
            self._umap_thumb_size: Tuple[int, int] = (0, 0)
            self._umap_ppu: float = 1.0

            # napari + points layers (created lazily in _update_scatter).
            self.points_layer = None
            self.lasso_layer = None
            self._point_size = 0.5

            # On first scatter draw we hide the host's existing layers (the
            # stitched mosaics and Finding Nemo's well-locations Points) so
            # the UMAP isn't competing with them for screen space. The prior
            # visibility is stashed here and restored in ``cleanup``.
            self._saved_layer_visibility: Dict[int, bool] = {}
            self._host_layers_hidden = False

            self._crop_full_pixmap = None
            self._assign_crop_labels: List[Tuple[QLabel, QLabel, Optional[QPixmap]]] = []

            self._build_ui()

        # ------------------------------------------------------------------
        # Public dock-widget contract
        # ------------------------------------------------------------------

        def as_dock_widget(self) -> "LabelTool":
            """Return the widget for ``FindingDory`` to dock.

            The widget IS the dock content — this just makes the contract
            explicit (replaces upstream ``run()`` which called
            ``napari.run()``; the parent app owns the event loop now).
            """
            return self

        def cleanup(self):
            """Restore the napari viewer to its pre-LabelTool state.

            Called by ``FindingDory.cleanup`` when the parent dock is being
            torn down. Restores host-layer visibility and removes the top
            toolbar dock so the user isn't left with orphaned widgets.
            """
            try:
                self._restore_host_layers()
            except Exception:
                log.exception("layer visibility restore failed")
            dock = getattr(self, "_toolbar_dock", None)
            if dock is not None:
                try:
                    self.viewer.window.remove_dock_widget(dock)
                except Exception:
                    log.exception("toolbar dock removal failed")
                self._toolbar_dock = None

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _scope(self) -> Tuple[str, str]:
            """Return current (fish_line, channel) scope."""
            return (self._current_line, self._current_channel)

        # ------------------------------------------------------------------
        # UI construction
        # ------------------------------------------------------------------

        def _build_ui(self):
            # The widget owns the layout. The napari viewer the parent
            # supplied is used only for the scatter + thumbnail layers.
            root_layout = QVBoxLayout(self)
            root_layout.setContentsMargins(4, 4, 4, 4)
            root_layout.setSpacing(4)

            # ── Toolbar (docked at napari top) ───────────────────────────
            toolbar = QWidget()
            toolbar_layout = QHBoxLayout(toolbar)
            toolbar_layout.setContentsMargins(2, 2, 2, 2)
            toolbar_layout.setSpacing(6)

            # Channel selector.
            toolbar_layout.addWidget(QLabel("Ch:"))
            self.channel_combo = QComboBox()
            self.channel_combo.addItems(self._all_channels)
            self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
            toolbar_layout.addWidget(self.channel_combo)

            # Color-by selector.
            toolbar_layout.addWidget(QLabel("Color:"))
            self.color_combo = QComboBox()
            self.color_combo.addItems(["group", "cluster"])
            self.color_combo.currentTextChanged.connect(self._on_color_changed)
            toolbar_layout.addWidget(self.color_combo)

            # Image UMAP toggle.
            self.image_umap_checkbox = QCheckBox("Show Fish")
            self.image_umap_checkbox.toggled.connect(self._toggle_image_umap)
            toolbar_layout.addWidget(self.image_umap_checkbox)

            # Crop size slider.
            toolbar_layout.addWidget(QLabel("Size:"))
            self._crop_size_slider = QSlider(Qt.Horizontal)
            self._crop_size_slider.setRange(10, 300)
            self._crop_size_slider.setValue(100)
            self._crop_size_slider.setToolTip(
                "Adjust fish crop size on UMAP (100% = median NN spacing)"
            )
            self._crop_size_slider.setMaximumWidth(100)
            self._crop_size_label = QLabel("100%")
            self._crop_size_label.setMinimumWidth(34)
            self._crop_size_slider.valueChanged.connect(self._on_crop_size_changed)
            toolbar_layout.addWidget(self._crop_size_slider)
            toolbar_layout.addWidget(self._crop_size_label)

            # Hide Assigned toggle.
            self._hide_assigned_checkbox = QCheckBox("Hide Assigned")
            self._hide_assigned_checkbox.setToolTip(
                "Hide wells already assigned to a group"
            )
            self._hide_assigned_checkbox.toggled.connect(self._toggle_hide_assigned)
            toolbar_layout.addWidget(self._hide_assigned_checkbox)

            # Recluster — re-runs UMAP + HDBSCAN on the current channel.
            # Existing assignments are preserved (auto-assign fires only on
            # the first clustering pass per scope; see _recompute_view).
            self.recluster_btn = QPushButton("Recluster")
            self.recluster_btn.setToolTip(
                "Re-run UMAP + HDBSCAN for the current channel. "
                "Existing group assignments are kept."
            )
            self.recluster_btn.clicked.connect(self._on_recluster)
            toolbar_layout.addWidget(self.recluster_btn)

            # Lasso toggle.
            self.lasso_btn = QPushButton("Lasso")
            self.lasso_btn.setCheckable(True)
            self.lasso_btn.toggled.connect(self._toggle_lasso)
            toolbar_layout.addWidget(self.lasso_btn)

            # Select-by attribute (ported from upstream). For Finding Dory the
            # only attributes that make sense are "Cluster" and "Group" — fluor
            # phenotype columns and picked-well info aren't part of this dock.
            toolbar_layout.addWidget(QLabel("Select by:"))
            self.select_by_combo = QComboBox()
            self.select_by_combo.addItem("(choose attribute)", "")
            self.select_by_combo.currentIndexChanged.connect(self._on_select_by_changed)
            toolbar_layout.addWidget(self.select_by_combo)

            self.select_value_combo = QComboBox()
            self.select_value_combo.setEnabled(False)
            toolbar_layout.addWidget(self.select_value_combo)

            self.select_all_btn = QPushButton("Select All")
            self.select_all_btn.setEnabled(False)
            self.select_all_btn.clicked.connect(self._on_select_by_value)
            toolbar_layout.addWidget(self.select_all_btn)

            toolbar_layout.addStretch()

            # Park the toolbar in napari's top dock area rather than embedding
            # it in the right-side panel — gives the controls full window width
            # and frees vertical space for the groups list + crop strip.
            # ``tabify=True`` matches the side panels: stacks behind any
            # existing top dock instead of splitting the row in half.
            self._toolbar_dock = self.viewer.window.add_dock_widget(
                toolbar, name="Finding Dory", area="top", tabify=True
            )

            # ── Groups panel ─────────────────────────────────────────────
            groups_panel = QWidget()
            groups_layout = QVBoxLayout(groups_panel)
            groups_layout.setContentsMargins(2, 2, 2, 2)
            groups_layout.setSpacing(2)

            self.group_list = QListWidget()
            self.group_list.setMaximumHeight(140)
            self.group_list.currentTextChanged.connect(self._on_group_focus_changed)
            self.group_list.itemDoubleClicked.connect(self._on_group_double_click)
            groups_layout.addWidget(self.group_list)

            grp_row = QHBoxLayout()
            grp_row.setSpacing(2)
            self.add_group_btn = QPushButton("+")
            self.add_group_btn.setToolTip("Add group")
            self.add_group_btn.clicked.connect(self._on_add_group)
            grp_row.addWidget(self.add_group_btn)
            self.rename_group_btn = QPushButton("Rename")
            self.rename_group_btn.setToolTip("Rename group")
            self.rename_group_btn.clicked.connect(self._on_rename_group)
            grp_row.addWidget(self.rename_group_btn)
            self.delete_group_btn = QPushButton("Del")
            self.delete_group_btn.setToolTip("Delete group")
            self.delete_group_btn.clicked.connect(self._on_delete_group)
            grp_row.addWidget(self.delete_group_btn)
            groups_layout.addLayout(grp_row)

            action_row = QHBoxLayout()
            action_row.setSpacing(2)
            self.assign_btn = QPushButton("Assign")
            self.assign_btn.setToolTip("Assign selected wells to group")
            self.assign_btn.setEnabled(False)
            self.assign_btn.clicked.connect(self._on_assign)
            action_row.addWidget(self.assign_btn)
            self.unassign_btn = QPushButton("Unassign")
            self.unassign_btn.setToolTip("Unassign selected wells")
            self.unassign_btn.setEnabled(False)
            self.unassign_btn.clicked.connect(self._on_unassign)
            action_row.addWidget(self.unassign_btn)
            groups_layout.addLayout(action_row)

            # Save — emits save_requested for FindingDory to handle.
            self.save_btn = QPushButton("Save")
            self.save_btn.setToolTip("Emit save_requested to FindingDory")
            self.save_btn.clicked.connect(self.save_requested.emit)
            groups_layout.addWidget(self.save_btn)

            self.status_label = QLabel("")
            self.status_label.setWordWrap(True)
            self.status_label.setStyleSheet("font-size: 10px; color: #aaa;")
            groups_layout.addWidget(self.status_label)

            root_layout.addWidget(groups_panel)

            # ── Selected well crop strip ─────────────────────────────────
            crop_widget = QWidget()
            crop_layout = QVBoxLayout(crop_widget)
            crop_layout.setContentsMargins(2, 2, 2, 2)
            crop_layout.setSpacing(2)

            self.crop_info = QLabel("Click a point or lasso-select")
            self.crop_info.setStyleSheet("font-weight: bold; font-size: 11px;")
            self.crop_info.setWordWrap(True)
            crop_layout.addWidget(self.crop_info)

            nav_row = QHBoxLayout()
            nav_row.setSpacing(4)
            self.prev_btn = QPushButton("<")
            self.prev_btn.setFixedWidth(28)
            self.prev_btn.setEnabled(False)
            self.prev_btn.clicked.connect(lambda: self._navigate_well(-1))
            nav_row.addWidget(self.prev_btn)
            self.nav_label = QLabel("")
            self.nav_label.setAlignment(Qt.AlignCenter)
            nav_row.addWidget(self.nav_label)
            self.next_btn = QPushButton(">")
            self.next_btn.setFixedWidth(28)
            self.next_btn.setEnabled(False)
            self.next_btn.clicked.connect(lambda: self._navigate_well(1))
            nav_row.addWidget(self.next_btn)
            crop_layout.addLayout(nav_row)

            self.crop_label = _ShrinkableLabel()
            self.crop_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.crop_label.setStyleSheet("background-color: #1a1a1a;")
            self.crop_label.setMinimumHeight(80)
            self.crop_label.resizeEvent = lambda e: self._update_crop_display()
            crop_layout.addWidget(self.crop_label)

            self.channel_legend_label = QLabel("")
            self.channel_legend_label.setStyleSheet("font-size: 10px; padding: 1px;")
            crop_layout.addWidget(self.channel_legend_label)
            root_layout.addWidget(crop_widget)

            # ── Assign cards (3-column grid of group thumbnails) ────────
            assign_widget = QWidget()
            assign_layout = QVBoxLayout(assign_widget)
            assign_layout.setContentsMargins(2, 2, 2, 2)
            assign_layout.setSpacing(0)

            self._assign_scroll = QScrollArea()
            self._assign_scroll.setWidgetResizable(True)
            self._assign_scroll.setFrameShape(QFrame.NoFrame)
            self._assign_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._assign_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            assign_layout.addWidget(self._assign_scroll)

            _orig_assign_resize = assign_widget.resizeEvent

            def _on_assign_resize(event):
                if _orig_assign_resize:
                    _orig_assign_resize(event)
                self._rescale_assign_crops()

            assign_widget.resizeEvent = _on_assign_resize
            root_layout.addWidget(assign_widget, 1)

            # Initialise the view for our single fish line.
            self._on_fish_line_changed(self._fish_line)

        # ------------------------------------------------------------------
        # Channel changes
        # ------------------------------------------------------------------

        def _on_fish_line_changed(self, line_name: str):
            """Set up the view for our single fish line.

            Upstream supported switching between fish lines; here it's
            invoked once at startup.
            """
            if not line_name:
                return
            self._remove_image_umap()
            self._current_line = line_name
            self._line_mask = (self.metadata["fish_line"] == line_name).values
            self._view_indices = np.where(self._line_mask)[0]
            if self.channel_combo.count() > 0:
                self._current_channel = self.channel_combo.currentText()
            self._recompute_view()

        def _on_channel_changed(self, channel: str):
            if not channel:
                return
            self._remove_image_umap()
            self._current_channel = channel
            self._recompute_view()
            if self._selected_well_list:
                wn, exp = self._selected_well_list[self._current_well_view_idx]
                self._show_crop(wn, exp)

        def _on_color_changed(self, color_by: str):
            self._color_by = color_by
            self._update_point_colors()

        def _on_recluster(self):
            """Re-run UMAP + HDBSCAN on the current channel.

            Existing assignments survive because the auto-assignment block
            in ``_recompute_view`` short-circuits on subsequent passes for a
            given (line, channel) scope (see ``self._auto_assigned``).
            """
            self._recompute_view()

        # ------------------------------------------------------------------
        # Select-by attribute → values dropdowns
        # ------------------------------------------------------------------

        def _refresh_select_by_combo(self):
            """Populate the "Select by" attribute combo.

            Finding Dory only exposes Cluster and Group; the upstream Fluor /
            Phenotype-combo / Picked options don't apply here (no per-well
            phenotype columns on ``self.metadata``, no ``_picked`` map).
            """
            if not hasattr(self, "select_by_combo"):
                return
            self.select_by_combo.blockSignals(True)
            self.select_by_combo.clear()
            self.select_by_combo.addItem("(choose attribute)", "")
            if self._view_clusters is not None:
                self.select_by_combo.addItem("Cluster", "cluster")
            self.select_by_combo.addItem("Group", "group")
            self.select_by_combo.blockSignals(False)
            self.select_value_combo.clear()
            self.select_value_combo.setEnabled(False)
            self.select_all_btn.setEnabled(False)

        def _on_select_by_changed(self, index: int):
            attr = self.select_by_combo.currentData()
            self.select_value_combo.clear()
            if not attr:
                self.select_value_combo.setEnabled(False)
                self.select_all_btn.setEnabled(False)
                return

            if attr == "cluster" and self._view_clusters is not None:
                unique_cl = sorted(
                    set(int(c) for c in self._view_clusters if c >= 0)
                )
                for cl in unique_cl:
                    n = int((self._view_clusters == cl).sum())
                    self.select_value_combo.addItem(f"cluster_{cl} ({n})", cl)
            elif attr == "group":
                fl, ch = self._scope()
                asgn = self.store.assignments(fl, ch)
                counts: Dict[str, int] = {}
                for g in asgn.values():
                    counts[g] = counts.get(g, 0) + 1
                for grp in self.store.groups(fl, ch):
                    n = counts.get(grp, 0)
                    self.select_value_combo.addItem(f"{grp} ({n})", grp)

            has_values = self.select_value_combo.count() > 0
            self.select_value_combo.setEnabled(has_values)
            self.select_all_btn.setEnabled(has_values)

        def _on_select_by_value(self):
            attr = self.select_by_combo.currentData()
            value = self.select_value_combo.currentData()
            if not attr or self._view_indices is None:
                return
            self._focused_group = None

            line_indices = self._view_indices
            if self._view_umap is not None:
                valid = ~np.isnan(self._view_umap).any(axis=1)
            else:
                valid = np.ones(len(line_indices), dtype=bool)
            for idx in self._hidden_indices:
                if 0 <= idx < len(valid):
                    valid[idx] = False

            selected: List[int] = []
            wells: List[Tuple[str, str]] = []

            if attr == "cluster" and self._view_clusters is not None:
                for li_pos in range(len(line_indices)):
                    if valid[li_pos] and int(self._view_clusters[li_pos]) == value:
                        selected.append(li_pos)
                        row = self.metadata.iloc[line_indices[li_pos]]
                        wells.append((row["well_name"], row["experiment"]))
            elif attr == "group":
                fl, ch = self._scope()
                asgn = self.store.assignments(fl, ch)
                for li_pos, meta_idx in enumerate(line_indices):
                    if not valid[li_pos]:
                        continue
                    row = self.metadata.iloc[meta_idx]
                    if asgn.get(row["well_id"]) == value:
                        selected.append(li_pos)
                        wells.append((row["well_name"], row["experiment"]))

            if not selected:
                self.crop_info.setText("No matching wells found.")
                return

            self._selected_indices = set(selected)
            self._highlight_selected(np.array(selected))
            seen = set()
            unique = []
            for w in wells:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            self._set_selected_wells(unique)
            self.assign_btn.setEnabled(True)
            self.unassign_btn.setEnabled(True)
            self.crop_info.setText(f"Selected {len(selected)} wells by {attr}")

        # ------------------------------------------------------------------
        # UMAP + clustering
        # ------------------------------------------------------------------

        def _get_channel_emb_for_line(
            self, channel: str, line_indices: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Return embeddings + valid line-positions for ``channel``."""
            if channel not in self.per_ch_idx or channel not in self.per_ch_emb:
                return np.zeros((0, 1)), np.array([], dtype=int)
            ch_idx_arr = self.per_ch_idx[channel]
            ch_emb = self.per_ch_emb[channel]
            ch_idx_set = {int(v): pos for pos, v in enumerate(ch_idx_arr)}

            valid_positions = []
            emb_rows = []
            for li_pos, meta_idx in enumerate(line_indices):
                pos = ch_idx_set.get(int(meta_idx))
                if pos is not None:
                    emb_rows.append(ch_emb[pos])
                    valid_positions.append(li_pos)

            if not emb_rows:
                return np.zeros((0, 1)), np.array([], dtype=int)
            return (
                np.asarray(emb_rows, dtype=np.float32),
                np.asarray(valid_positions, dtype=int),
            )

        def _recompute_view(self):
            """Recompute UMAP + clustering for the current channel.

            Refactor item 7: clustering routes through the injected
            ``ClusterStrategy`` instead of constructing HDBSCAN directly.
            """
            if self._view_indices is None or len(self._view_indices) < 2:
                self.status_label.setText("Too few wells for this fish line.")
                return

            line_indices = self._view_indices
            n = len(line_indices)

            embeddings, valid_positions = self._get_channel_emb_for_line(
                self._current_channel, line_indices
            )
            if len(embeddings) < 2:
                self.status_label.setText(
                    f"Too few wells with channel {self._current_channel}."
                )
                return

            mask = np.zeros(n, dtype=bool)
            mask[valid_positions] = True
            self._view_valid_mask = mask
            full_emb = np.full((n, embeddings.shape[1]), np.nan, dtype=np.float32)
            full_emb[valid_positions] = embeddings
            self._view_embeddings = full_emb

            valid_emb = full_emb[mask]
            if len(valid_emb) < 2:
                self.status_label.setText("Too few valid embeddings.")
                return

            # Refactor item 7: cluster via the injected strategy.
            try:
                valid_labels = np.asarray(self._cluster_strategy.cluster(valid_emb))
            except Exception as e:
                log.exception("cluster strategy failed")
                self.status_label.setText(f"Clustering failed: {e}")
                valid_labels = np.full(len(valid_emb), -1, dtype=int)

            full_labels = np.full(n, -2, dtype=int)
            full_labels[mask] = valid_labels
            self._view_clusters = full_labels

            # UMAP — lazy import.
            try:
                from umap import UMAP

                n_neighbors = min(15, len(valid_emb) - 1)
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=42,
                )
                umap_2d = reducer.fit_transform(valid_emb).astype(np.float32)
            except Exception as e:
                log.exception("UMAP failed")
                self.status_label.setText(f"UMAP failed: {e}")
                return

            full_umap = np.full((n, 2), np.nan, dtype=np.float32)
            full_umap[mask] = umap_2d
            self._view_umap = full_umap

            # Auto-create starting groups from cluster labels. The first time
            # we cluster a given (line, channel) scope we *also* populate each
            # cluster_N group with its members so the UI shows real counts.
            # Subsequent reclusters (via the Recluster button) recompute UMAP
            # + labels but never overwrite assignments — manual edits and
            # already-populated cluster rows survive.
            fl, ch = self._scope()
            asgn = self.store.assignments(fl, ch)
            unique_clusters = sorted(set(int(c) for c in valid_labels if c >= 0))
            first_time = (fl, ch) not in self._auto_assigned
            cluster_members: Dict[int, List[str]] = {cl: [] for cl in unique_clusters}
            if first_time:
                for li_pos, meta_idx in enumerate(line_indices):
                    if not mask[li_pos]:
                        continue
                    cl = int(full_labels[li_pos])
                    if cl < 0:
                        continue
                    wid = self.metadata.iloc[meta_idx]["well_id"]
                    if wid in asgn:
                        continue
                    cluster_members[cl].append(wid)
            for cl in unique_clusters:
                group_name = f"cluster_{cl}"
                self.store.create_group(fl, ch, group_name)
                if first_time and cluster_members[cl]:
                    self.store.assign(fl, ch, cluster_members[cl], group_name)
            if first_time:
                self._auto_assigned.add((fl, ch))

            self._refresh_group_list()
            self._refresh_select_by_combo()
            self._preload_crops()
            self._update_scatter()
            self._update_status()

            if self.image_umap_checkbox.isChecked():
                self._render_image_umap()

        # ------------------------------------------------------------------
        # Scatter display
        # ------------------------------------------------------------------

        def _update_scatter(self):
            if self._view_umap is None:
                return

            coords = self._view_umap.copy()
            valid = ~np.isnan(coords).any(axis=1)
            valid_coords = coords[valid]
            if len(valid_coords) > 0:
                rx = float(valid_coords[:, 0].max() - valid_coords[:, 0].min())
                ry = float(valid_coords[:, 1].max() - valid_coords[:, 1].min())
                data_range = max(rx, ry, 1e-6)
                self._point_size = data_range / 80
            else:
                self._point_size = 0.5

            colors = self._build_colors()

            # napari Points use (y, x) order.
            display_coords = np.column_stack([coords[:, 1], coords[:, 0]])

            if self.points_layer is None:
                self.points_layer = self.viewer.add_points(
                    display_coords,
                    face_color=colors,
                    size=self._point_size,
                    border_width=0,
                    name="UMAP",
                )
                self.points_layer.mouse_drag_callbacks.append(self._on_point_click)
                self._hide_host_layers()
                self._frame_camera_on_umap(display_coords[valid])
            else:
                self.points_layer.data = display_coords
                self.points_layer.face_color = colors
                self.points_layer.size = self._point_size

            self._selected_indices = set()
            self.assign_btn.setEnabled(False)
            self.unassign_btn.setEnabled(False)

            if self.lasso_layer is not None:
                self.lasso_layer.data = []

            self._umap_ref_points_data = display_coords.copy()
            valid_display = display_coords[valid]
            if len(valid_display) > 0:
                self._umap_world_centroid = (
                    float(valid_display[:, 0].mean()),
                    float(valid_display[:, 1].mean()),
                )
            if hasattr(self, "_crop_size_slider"):
                self._crop_size_slider.blockSignals(True)
                self._crop_size_slider.setValue(100)
                self._crop_size_slider.blockSignals(False)
                self._crop_size_label.setText("100%")

        def _hide_host_layers(self):
            """Hide pre-existing layers so the UMAP gets the whole canvas.

            Keyed by ``id(layer)`` because layer names can collide and
            ``viewer.layers`` order can shift. Idempotent.
            """
            if self._host_layers_hidden:
                return
            for layer in list(self.viewer.layers):
                if layer is self.points_layer:
                    continue
                self._saved_layer_visibility[id(layer)] = bool(layer.visible)
                layer.visible = False
            self._host_layers_hidden = True

        def _restore_host_layers(self):
            """Restore the visibility we stashed in ``_hide_host_layers``."""
            if not self._host_layers_hidden:
                return
            for layer in list(self.viewer.layers):
                if layer is self.points_layer:
                    continue
                prev = self._saved_layer_visibility.get(id(layer))
                if prev is not None:
                    layer.visible = prev
            self._saved_layer_visibility.clear()
            self._host_layers_hidden = False

        def _frame_camera_on_umap(self, valid_display: np.ndarray):
            """Center & zoom the napari camera on the UMAP scatter bounds."""
            if len(valid_display) == 0:
                return
            try:
                y_min, x_min = valid_display.min(axis=0)
                y_max, x_max = valid_display.max(axis=0)
                cy = float((y_min + y_max) / 2.0)
                cx = float((x_min + x_max) / 2.0)
                extent = float(max(y_max - y_min, x_max - x_min, 1e-6))
                # Pad ~10% so points at the edge aren't clipped.
                # Canvas size isn't known here; napari treats zoom as
                # pixels-per-data-unit, so we approximate with a typical
                # 800-px canvas and let the user zoom afterwards.
                self.viewer.camera.center = (cy, cx)
                self.viewer.camera.zoom = 800.0 / (extent * 1.2)
            except Exception:
                log.exception("could not frame camera on UMAP")

        def _build_colors(self):
            n = len(self._view_umap)
            colors = np.full((n, 4), _UNASSIGNED_COLOR, dtype=np.float32)
            valid = ~np.isnan(self._view_umap).any(axis=1)
            colors[~valid, 3] = 0.0

            line_indices = self._view_indices
            if line_indices is None:
                return colors

            if (
                hasattr(self, "_hide_assigned_checkbox")
                and self._hide_assigned_checkbox.isChecked()
            ):
                fl, ch = self._scope()
                assigned_wids = set(self.store.assignments(fl, ch).keys())
                self._hidden_indices = set()
                for li_pos, meta_idx in enumerate(line_indices):
                    wid = self.metadata.iloc[meta_idx]["well_id"]
                    if wid in assigned_wids:
                        self._hidden_indices.add(li_pos)
            if self._hidden_indices:
                for idx in self._hidden_indices:
                    if 0 <= idx < n:
                        colors[idx, 3] = 0.0

            fl, ch = self._scope()

            if self._color_by == "group":
                asgn = self.store.assignments(fl, ch)
                for li_pos, meta_idx in enumerate(line_indices):
                    if not valid[li_pos]:
                        continue
                    wid = self.metadata.iloc[meta_idx]["well_id"]
                    group = asgn.get(wid)
                    if group:
                        colors[li_pos] = self.store.group_color(fl, ch, group)
                    else:
                        colors[li_pos] = _UNASSIGNED_COLOR

            elif self._color_by == "cluster":
                if self._view_clusters is not None:
                    unique_cl = sorted(set(self._view_clusters) - {-1, -2})
                    for li_pos in range(n):
                        if not valid[li_pos]:
                            continue
                        cl = int(self._view_clusters[li_pos])
                        if cl == -1:
                            colors[li_pos] = _NOISE_COLOR
                        elif cl == -2:
                            colors[li_pos, 3] = 0.0
                        else:
                            ci = unique_cl.index(cl) % len(_TAB20)
                            colors[li_pos] = _TAB20[ci]

            return colors

        def _update_point_colors(self):
            if self.points_layer is None:
                return
            colors = self._build_colors()
            self.points_layer.face_color = colors

        # ------------------------------------------------------------------
        # Point click
        # ------------------------------------------------------------------

        def _on_point_click(self, layer, event):
            if self._lasso_mode:
                return
            if event.type != "mouse_press":
                return
            if self.points_layer is None or self._view_umap is None:
                return

            coords = self.points_layer.data
            click_pos = np.array(event.position[:2])
            valid = ~np.isnan(self._view_umap).any(axis=1)
            for idx in self._hidden_indices:
                if 0 <= idx < len(valid):
                    valid[idx] = False
            dists = np.full(len(coords), np.inf)
            dists[valid] = np.linalg.norm(coords[valid] - click_pos, axis=1)

            nearest = int(np.argmin(dists))
            if dists[nearest] > self._point_size * 3:
                return

            meta_idx = self._view_indices[nearest]
            row = self.metadata.iloc[meta_idx]
            self._selected_indices = {nearest}
            self._set_selected_wells([(row["well_name"], row["experiment"])])
            self._highlight_selected(np.array([nearest]))
            self.assign_btn.setEnabled(True)
            self.unassign_btn.setEnabled(True)

            fl, ch = self._scope()
            wid = row["well_id"]
            group = self.store.assignments(fl, ch).get(wid, "unassigned")
            self.crop_info.setText(
                f"{row['well_name']} | {row['experiment']} | {group}"
            )

        # ------------------------------------------------------------------
        # Lasso selection
        # ------------------------------------------------------------------

        def _toggle_lasso(self, enabled: bool):
            self._lasso_mode = enabled
            self._ignore_lasso_event = False
            if enabled:
                self.lasso_btn.setText("Lasso ON")
                if self.lasso_layer is None:
                    self.lasso_layer = self.viewer.add_shapes(
                        name="Lasso",
                        shape_type="polygon",
                        edge_color="yellow",
                        edge_width=0,
                        face_color=[1, 1, 0, 0.1],
                    )
                    self.lasso_layer.events.data.connect(self._on_lasso_data_changed)
                self.lasso_layer.data = []
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(self.lasso_layer)
                self.lasso_layer.mode = "add_polygon"
            else:
                self.lasso_btn.setText("Lasso")
                if self.lasso_layer is not None:
                    self.lasso_layer.data = []
                self.viewer.layers.selection.clear()
                if self.points_layer is not None:
                    self.viewer.layers.selection.add(self.points_layer)

        def _rearm_lasso(self):
            """Re-arm the lasso for another draw.

            Does NOT clear the old shape — napari crashes if you set
            ``.data=[]`` while ``mode='add_polygon'``. ``_process_lasso``
            always uses ``data[-1]`` so leftover shapes are harmless.
            """
            if not self._lasso_mode or self.lasso_layer is None:
                return
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(self.lasso_layer)
            try:
                self.lasso_layer.mode = "add_polygon"
            except Exception:
                pass

        def _on_lasso_data_changed(self, event):
            if getattr(self, "_ignore_lasso_event", False):
                return
            if not self._lasso_mode:
                return
            if self.lasso_layer is not None and len(self.lasso_layer.data) > 0:
                self._process_lasso()

        def _process_lasso(self):
            if self.lasso_layer is None or len(self.lasso_layer.data) == 0:
                return

            from matplotlib.path import Path as MplPath

            polygon = self.lasso_layer.data[-1]
            if len(polygon) < 3:
                return

            path = MplPath(polygon[:, :2])

            coords = self.points_layer.data
            valid = ~np.isnan(self._view_umap).any(axis=1)
            for idx in self._hidden_indices:
                if 0 <= idx < len(valid):
                    valid[idx] = False

            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                self.crop_info.setText("No points in lasso.")
                self._rearm_lasso()
                return

            inside = path.contains_points(coords[valid_indices])
            selected = valid_indices[inside]

            if len(selected) == 0:
                self.crop_info.setText("No points in lasso.")
                self._rearm_lasso()
                return

            self._focused_group = None
            # Capture into selection state — Assign/Unassign commit.
            self._selected_indices = set(selected.tolist())
            self._highlight_selected(selected)

            wells = []
            for li_pos in selected:
                meta_idx = self._view_indices[li_pos]
                row = self.metadata.iloc[meta_idx]
                wells.append((row["well_name"], row["experiment"]))
            seen = set()
            unique = []
            for w in wells:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            self._set_selected_wells(unique)
            self.assign_btn.setEnabled(True)
            self.unassign_btn.setEnabled(True)
            self._refresh_group_list()
            self._update_status()

            self.crop_info.setText(f"Selected {len(selected)} wells — Assign or Unassign")
            self._rearm_lasso()

        def _highlight_selected(self, indices: np.ndarray):
            if self.points_layer is None:
                return
            colors = self._build_colors()
            colors[indices] = [1.0, 1.0, 0.0, 1.0]
            self.points_layer.face_color = colors

        # ------------------------------------------------------------------
        # Crop display
        # ------------------------------------------------------------------

        def _contrast_for(self, channel: str) -> Tuple[Optional[float], Optional[float]]:
            """Return ``(low, high)`` for ``channel`` or ``(None, None)``.

            ``None`` falls through to per-crop percentile inside
            ``_uint16_to_rgb`` — that's the per-crop overexposure path, used
            only when no host contrast was supplied (e.g. the mosaic was
            renamed between Finding Nemo and Finding Dory).
            """
            bounds = self._channel_contrast.get(channel)
            if bounds is None:
                return (None, None)
            return (float(bounds[0]), float(bounds[1]))

        def _preload_crops(self):
            """Build RGB crops for the active channel from ``self._well_crops``.

            Refactor item 4: no more per-experiment ``WellLoader.mosaics``
            disk reads — crops are already cropped numpy arrays.
            """
            self._crop_cache.clear()
            if self._view_indices is None:
                return

            active_channel = self._current_channel
            display_cfg = get_channel_display(active_channel)
            rgb_color = display_cfg.rgb_color
            dark_on_white = active_channel.upper() in ("DAPI", "CY5")
            low, high = self._contrast_for(active_channel)

            loaded = 0
            for meta_idx in self._view_indices:
                widx = int(meta_idx)
                row = self.metadata.iloc[widx]
                wn, exp = row["well_name"], row["experiment"]
                if (wn, exp) in self._crop_cache:
                    continue
                if widx >= len(self._well_crops):
                    continue
                crops_for_well = self._well_crops[widx]
                crop = crops_for_well.get(active_channel)
                if crop is None or crop.size == 0:
                    continue
                try:
                    rgb = _uint16_to_rgb(
                        crop, rgb_color, low=low, high=high, dark_on_white=dark_on_white,
                    )
                    self._crop_cache[(wn, exp)] = rgb
                    loaded += 1
                except Exception as e:
                    log.debug(f"preload skipped well {wn}|{active_channel}: {e}")

            log.info(
                f"Preloaded {loaded} crops for {self._current_line}|{active_channel}"
            )

        def _set_selected_wells(self, wells: List[Tuple[str, str]]):
            self._selected_well_list = wells
            self._current_well_view_idx = 0
            has_nav = len(wells) > 1
            self.prev_btn.setEnabled(has_nav)
            self.next_btn.setEnabled(has_nav)
            if wells:
                self._update_nav_label()
                self._show_crop(wells[0][0], wells[0][1])
            else:
                self.nav_label.setText("")
                self.crop_label.clear()
                self._crop_full_pixmap = None
                self.channel_legend_label.setText("")

        def _navigate_well(self, direction: int):
            if not self._selected_well_list:
                return
            self._current_well_view_idx = (
                (self._current_well_view_idx + direction)
                % len(self._selected_well_list)
            )
            self._update_nav_label()
            wn, exp = self._selected_well_list[self._current_well_view_idx]
            self._show_crop(wn, exp)

        def _update_nav_label(self):
            n = len(self._selected_well_list)
            i = self._current_well_view_idx + 1
            self.nav_label.setText(f"{i} of {n}")

        def _show_crop(self, well_name: str, experiment: str):
            key = (well_name, experiment)
            rgb = self._crop_cache.get(key)

            if rgb is None:
                # On-demand: look up the well's uint16 crop and render.
                match = self.metadata[
                    (self.metadata["well_name"] == well_name)
                    & (self.metadata["experiment"] == experiment)
                ]
                if match.empty:
                    self.crop_label.setText(f"No metadata row for {well_name}")
                    return
                widx = int(match.index[0])
                if widx >= len(self._well_crops):
                    self.crop_label.setText(f"No crop for well index {widx}")
                    return
                crops_for_well = self._well_crops[widx]
                active_channel = self._current_channel
                crop = crops_for_well.get(active_channel)
                if crop is None or crop.size == 0:
                    self.crop_label.setText(
                        f"No crop for {active_channel} on well {well_name}"
                    )
                    return
                display_cfg = get_channel_display(active_channel)
                dark_on_white = active_channel.upper() in ("DAPI", "CY5")
                low, high = self._contrast_for(active_channel)
                try:
                    rgb = _uint16_to_rgb(
                        crop, display_cfg.rgb_color,
                        low=low, high=high, dark_on_white=dark_on_white,
                    )
                    self._crop_cache[key] = rgb
                except Exception as e:
                    self.crop_label.setText(f"Render error: {e}")
                    return

            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            self._crop_full_pixmap = QPixmap.fromImage(qimg)
            self._update_crop_display()

            # Legend strip — active channel highlighted, others dim.
            legend_parts = []
            for ch_name in self._all_channels:
                cfg = get_channel_display(ch_name)
                ri, gi, bi = [int(v * 255) for v in cfg.rgb_color]
                active = ch_name == self._current_channel
                style = f"color:rgb({ri},{gi},{bi})" if active else "color:#555"
                legend_parts.append(
                    f'<span style="{style}">[#]</span> {ch_name}'
                )
            self.channel_legend_label.setText("  ".join(legend_parts))

        def _update_crop_display(self):
            if getattr(self, "_crop_updating", False):
                return
            self._crop_updating = True
            try:
                if self._crop_full_pixmap is None or self._crop_full_pixmap.isNull():
                    self.crop_label.clear()
                    return
                parent = self.crop_label.parent()
                w = (parent.width() - 12) if parent else self.crop_label.width()
                if w < 1:
                    return
                scaled = self._crop_full_pixmap.scaledToWidth(w, Qt.SmoothTransformation)
                self.crop_label.setPixmap(scaled)
            finally:
                self._crop_updating = False

        # ------------------------------------------------------------------
        # Group manager
        # ------------------------------------------------------------------

        def _refresh_group_list(self):
            self.group_list.blockSignals(True)
            current = self.group_list.currentItem()
            current_text = current.text() if current else None
            self.group_list.clear()

            fl, ch = self._scope()
            counts = self.store.counts(fl, ch)
            for g in self.store.groups(fl, ch):
                c = counts.get(g, 0)
                self.group_list.addItem(f"{g} ({c})" if c else g)
            # Virtual "unassigned" entry.
            n_assigned = sum(counts.values())
            n_total = len(self._view_indices) if self._view_indices is not None else 0
            n_unassigned = n_total - n_assigned
            self.group_list.addItem(f"unassigned ({n_unassigned})")
            if current_text:
                for i in range(self.group_list.count()):
                    item_text = self.group_list.item(i).text()
                    if item_text == current_text or item_text.startswith(current_text + " ("):
                        self.group_list.setCurrentRow(i)
                        break
            self.group_list.blockSignals(False)
            if hasattr(self, "_assign_scroll"):
                self._rebuild_quick_assign_buttons()
            # Keep the Select-by value combo in sync when "Group" is active.
            if hasattr(self, "select_by_combo") and self.select_by_combo.currentData() == "group":
                self._on_select_by_changed(self.select_by_combo.currentIndex())

        def _rebuild_quick_assign_buttons(self):
            """Build the 3-column grid of group cards with thumbnails.

            Row order: globals (empty/multiple/deformed) bold at the top,
            then custom groups, unassign always last.
            """
            fl, ch = self._scope()
            counts = self.store.counts(fl, ch)
            all_groups = self.store.groups(fl, ch)

            pinned = [g for g in all_groups if g in GLOBAL_GROUPS]
            custom = [g for g in all_groups if g not in GLOBAL_GROUPS]
            ordered = pinned + custom

            cols = 3
            inner = QWidget()
            grid = QGridLayout(inner)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            grid.setAlignment(Qt.AlignTop)
            for c in range(cols):
                grid.setColumnStretch(c, 1)

            self._assign_crop_labels = []

            def _make_card(label_text, r, g, b, text_color, click_fn, thumb, bold=False):
                card = QWidget()
                card_lay = QVBoxLayout(card)
                card_lay.setContentsMargins(0, 0, 0, 0)
                card_lay.setSpacing(0)
                card_lay.setAlignment(Qt.AlignTop)
                card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                card.setCursor(Qt.PointingHandCursor)
                card.mousePressEvent = click_fn

                weight = "bold" if bold else "normal"
                title = QLabel(label_text)
                title.setStyleSheet(
                    f"background-color: rgb({r},{g},{b}); color: {text_color}; "
                    f"font-size: 10px; font-weight: {weight}; padding: 1px 4px;"
                )
                title.setCursor(Qt.PointingHandCursor)
                title.mousePressEvent = click_fn
                card_lay.addWidget(title)

                crop_lbl = QLabel()
                crop_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                crop_lbl.setStyleSheet("background: #111;")
                crop_lbl.setCursor(Qt.PointingHandCursor)
                crop_lbl.mousePressEvent = click_fn
                card_lay.addWidget(crop_lbl)

                full_pm = None
                if thumb is not None:
                    th, tw = thumb.shape[:2]
                    qimg = QImage(thumb.data, tw, th, tw * 3, QImage.Format_RGB888)
                    full_pm = QPixmap.fromImage(qimg)

                self._assign_crop_labels.append((title, crop_lbl, full_pm))
                return card

            idx = 0
            for group_name in ordered:
                color = self.store.group_color(fl, ch, group_name)
                r = int(color[0] * 255)
                g = int(color[1] * 255)
                b = int(color[2] * 255)
                lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = "#000" if lum > 0.5 else "#fff"
                count = counts.get(group_name, 0)
                click_fn = lambda event, gn=group_name: self._quick_assign(gn)
                thumb = self._get_group_thumbnail(group_name)
                bold = group_name in GLOBAL_GROUPS
                lock = " [L]" if bold else ""
                card = _make_card(
                    f" {group_name} ({count}){lock}",
                    r, g, b, text_color, click_fn, thumb, bold=bold,
                )
                grid.addWidget(card, idx // cols, idx % cols)
                idx += 1

            unassign_card = _make_card(
                " Unassign", 68, 68, 68, "#ccc",
                lambda event: self._quick_unassign(), None,
            )
            grid.addWidget(unassign_card, idx // cols, idx % cols)

            self._assign_scroll.setWidget(inner)
            self._rescale_assign_crops()

        def _rescale_assign_crops(self):
            if getattr(self, "_assign_rescaling", False):
                return
            self._assign_rescaling = True
            try:
                self._rescale_assign_crops_inner()
            finally:
                self._assign_rescaling = False

        def _rescale_assign_crops_inner(self):
            vp = self._assign_scroll.viewport().width() or 200
            n_cols = 3
            col_w = max(40, (vp - 16) // n_cols)
            for title_lbl, crop_lbl, full_pm in getattr(self, "_assign_crop_labels", []):
                w = col_w
                # Square-ish cards — height tracks width with a fish-like aspect.
                h = max(1, int(w * 416 / 1808))
                if full_pm is not None and not full_pm.isNull():
                    scaled = full_pm.scaledToWidth(w, Qt.SmoothTransformation)
                    crop_lbl.setPixmap(scaled)
                else:
                    crop_lbl.clear()
                crop_lbl.setFixedHeight(h)
                title_lbl.setFixedHeight(h)

        def _get_group_thumbnail(self, group_name: str) -> Optional[np.ndarray]:
            """Representative RGB crop for the group — first member found.

            Refactor item 4: looks up via ``self._well_crops`` instead of
            ``self._well_loaders[exp].get_well_crop(...)``.
            """
            fl, ch = self._scope()
            members = self.store.get_group_members(fl, ch, group_name)
            if not members:
                return None
            for wid in members:
                widx = self._wid_to_widx.get(wid)
                if widx is None or widx >= len(self._well_crops):
                    continue
                row = self.metadata.iloc[widx]
                rgb = self._crop_cache.get((row["well_name"], row["experiment"]))
                if rgb is not None:
                    return rgb
                # Cache miss — render the active channel on the fly.
                crop = self._well_crops[widx].get(self._current_channel)
                if crop is None or crop.size == 0:
                    continue
                try:
                    cfg = get_channel_display(self._current_channel)
                    dark_on_white = self._current_channel.upper() in ("DAPI", "CY5")
                    low, high = self._contrast_for(self._current_channel)
                    return _uint16_to_rgb(
                        crop, cfg.rgb_color,
                        low=low, high=high, dark_on_white=dark_on_white,
                    )
                except Exception:
                    continue
            return None

        def _quick_assign(self, group_name: str):
            """Assign the currently viewed well to ``group_name`` and advance."""
            if not self._selected_well_list:
                return
            wn, exp = self._selected_well_list[self._current_well_view_idx]
            match = self.metadata[
                (self.metadata["experiment"] == exp) & (self.metadata["well_name"] == wn)
            ]
            if match.empty:
                return
            wid = match.iloc[0]["well_id"]
            fl, ch = self._scope()
            self.store.assign(fl, ch, [wid], group_name)
            self.crop_info.setText(f"{wn} | {exp} -> {group_name}")
            self._advance_after_action()

        def _quick_unassign(self):
            if not self._selected_well_list:
                return
            wn, exp = self._selected_well_list[self._current_well_view_idx]
            match = self.metadata[
                (self.metadata["experiment"] == exp) & (self.metadata["well_name"] == wn)
            ]
            if match.empty:
                return
            wid = match.iloc[0]["well_id"]
            fl, ch = self._scope()
            self.store.unassign(fl, ch, [wid])
            self.crop_info.setText(f"{wn} | {exp} -> unassigned")
            self._advance_after_action()

        def _advance_after_action(self):
            if not self._selected_well_list:
                return
            self._selected_well_list.pop(self._current_well_view_idx)
            if not self._selected_well_list:
                self._current_well_view_idx = 0
                self.nav_label.setText("")
                self.prev_btn.setEnabled(False)
                self.next_btn.setEnabled(False)
                self.crop_info.setText("All wells processed.")
            else:
                if self._current_well_view_idx >= len(self._selected_well_list):
                    self._current_well_view_idx = 0
                self._update_nav_label()
                wn, exp = self._selected_well_list[self._current_well_view_idx]
                self._show_crop(wn, exp)
                has_nav = len(self._selected_well_list) > 1
                self.prev_btn.setEnabled(has_nav)
                self.next_btn.setEnabled(has_nav)
            self._refresh_group_list()
            self._update_point_colors()

        def _get_selected_group_name(self) -> Optional[str]:
            item = self.group_list.currentItem()
            if item is None:
                return None
            text = item.text()
            if " (" in text:
                text = text.rsplit(" (", 1)[0]
            return text

        def _on_group_focus_changed(self, text: str):
            fl, ch = self._scope()
            group = self._get_selected_group_name()
            if group:
                members = self.store.get_group_members(fl, ch, group)
                self.status_label.setText(f"Focused: {group} ({len(members)} wells)")

        def _on_group_double_click(self, item):
            text = item.text()
            if " (" in text:
                text = text.rsplit(" (", 1)[0]
            group = text

            if self._focused_group == group:
                self._deselect_all()
                return

            fl, ch = self._scope()

            if group == "unassigned":
                assigned_wids = set(self.store.assignments(fl, ch).keys())
                indices = []
                wells = []
                for li_pos, meta_idx in enumerate(self._view_indices):
                    wid = self.metadata.iloc[meta_idx]["well_id"]
                    if wid not in assigned_wids:
                        valid = self._view_valid_mask
                        if valid is not None and not valid[li_pos]:
                            continue
                        indices.append(li_pos)
                        row = self.metadata.iloc[meta_idx]
                        wells.append((row["well_name"], row["experiment"]))
            else:
                member_wids = set(self.store.get_group_members(fl, ch, group))
                if not member_wids:
                    self.crop_info.setText(f"'{group}' has no members.")
                    return
                indices = []
                wells = []
                for li_pos, meta_idx in enumerate(self._view_indices):
                    wid = self.metadata.iloc[meta_idx]["well_id"]
                    if wid in member_wids:
                        indices.append(li_pos)
                        row = self.metadata.iloc[meta_idx]
                        wells.append((row["well_name"], row["experiment"]))

            if not indices:
                self.crop_info.setText(f"'{group}' members not visible in current view.")
                return

            self._focused_group = group
            selected = np.array(indices)
            self._selected_indices = set(indices)
            self._highlight_selected(selected)

            seen = set()
            unique = []
            for w in wells:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            self._set_selected_wells(unique)
            self.assign_btn.setEnabled(True)
            self.unassign_btn.setEnabled(True)
            self.crop_info.setText(f"'{group}': {len(indices)} wells highlighted")

        def _deselect_all(self):
            self._focused_group = None
            self._selected_indices = set()
            self.assign_btn.setEnabled(False)
            self.unassign_btn.setEnabled(False)
            self._update_point_colors()
            self._set_selected_wells([])
            self.crop_info.setText("No selection")

        def _on_add_group(self):
            name, ok = QInputDialog.getText(self, "New Group", "Group name:")
            if ok and name.strip():
                fl, ch = self._scope()
                if not self.store.create_group(fl, ch, name.strip()):
                    QMessageBox.warning(self, "Group exists", f"'{name}' already exists.")
                    return
                self._refresh_group_list()

        def _on_rename_group(self):
            old = self._get_selected_group_name()
            if not old or old in GLOBAL_GROUPS:
                QMessageBox.warning(
                    self, "Can't rename", "Default groups can't be renamed."
                )
                return
            new_name, ok = QInputDialog.getText(
                self, "Rename Group", "New name:", text=old
            )
            if ok and new_name.strip():
                fl, ch = self._scope()
                if not self.store.rename_group(fl, ch, old, new_name.strip()):
                    QMessageBox.warning(self, "Rename failed", "Name conflict or unknown group.")
                    return
                self._refresh_group_list()
                self._update_point_colors()

        def _on_delete_group(self):
            group = self._get_selected_group_name()
            if not group or group in GLOBAL_GROUPS:
                QMessageBox.warning(
                    self, "Can't delete", "Default groups can't be deleted."
                )
                return
            fl, ch = self._scope()
            count = len(self.store.get_group_members(fl, ch, group))
            reply = QMessageBox.question(
                self, "Delete Group",
                f"Delete '{group}' and unassign {count} wells?",
            )
            if reply == QMessageBox.Yes:
                self.store.delete_group(fl, ch, group)
                self._refresh_group_list()
                self._update_point_colors()
                self._update_status()

        def _on_assign(self):
            """Commit current selection to the chosen group.

            Verify-before-commit: ``_selected_indices`` was populated by the
            lasso or click handlers; this button only fires the assignment.
            """
            group = self._get_selected_group_name()
            if not group:
                QMessageBox.warning(
                    self, "No Group Selected", "Select a group in the list first.",
                )
                return

            fl, ch = self._scope()
            well_ids = []
            for li_pos in self._selected_indices:
                meta_idx = self._view_indices[li_pos]
                wid = self.metadata.iloc[meta_idx]["well_id"]
                well_ids.append(wid)

            self.store.assign(fl, ch, well_ids, group)
            n = len(well_ids)
            self._selected_indices = set()
            self._focused_group = None
            self._refresh_group_list()
            self._update_point_colors()
            self._update_status()
            self._set_selected_wells([])
            self._rearm_lasso()
            self.crop_info.setText(f"Assigned {n} wells to '{group}'")

        def _on_unassign(self):
            fl, ch = self._scope()
            well_ids = []
            for li_pos in self._selected_indices:
                meta_idx = self._view_indices[li_pos]
                wid = self.metadata.iloc[meta_idx]["well_id"]
                well_ids.append(wid)

            self.store.unassign(fl, ch, well_ids)
            n = len(well_ids)
            self._selected_indices = set()
            self._focused_group = None
            self._refresh_group_list()
            self._update_point_colors()
            self._update_status()
            self._set_selected_wells([])
            self._rearm_lasso()
            self.crop_info.setText(f"Unassigned {n} wells")

        # ------------------------------------------------------------------
        # Status
        # ------------------------------------------------------------------

        def _update_status(self):
            fl, ch = self._scope()
            total = len(self._view_indices) if self._view_indices is not None else 0

            asgn = self.store.assignments(fl, ch)
            indices = self._view_indices if self._view_indices is not None else []
            assigned = sum(
                1 for i in indices if self.metadata.iloc[i]["well_id"] in asgn
            )
            counts = self.store.counts(fl, ch)
            lines = [
                f"Scope: {fl} | {ch}",
                f"Wells: {assigned}/{total} assigned",
            ]
            for g in self.store.groups(fl, ch):
                c = counts.get(g, 0)
                if c > 0:
                    lines.append(f"  {g}: {c}")
            self.status_label.setText("\n".join(lines))

        # ------------------------------------------------------------------
        # Image UMAP (thumbnails-on-scatter overlay)
        # ------------------------------------------------------------------

        def _toggle_hide_assigned(self, checked: bool):
            if checked:
                fl, ch = self._scope()
                assigned_wids = set(self.store.assignments(fl, ch).keys())
                self._hidden_indices = set()
                if self._view_indices is not None:
                    for li_pos, meta_idx in enumerate(self._view_indices):
                        wid = self.metadata.iloc[meta_idx]["well_id"]
                        if wid in assigned_wids:
                            self._hidden_indices.add(li_pos)
            else:
                self._hidden_indices.clear()
            self._update_point_colors()
            if self.image_umap_checkbox.isChecked():
                self._render_image_umap()

        def _toggle_image_umap(self, checked: bool):
            if checked:
                self._render_image_umap()
            else:
                self._remove_image_umap()

        def _on_crop_size_changed(self, value: int):
            """Spread UMAP points + rebuild thumbnail canvas to match."""
            self._crop_size_label.setText(f"{value}%")
            if self.points_layer is None or self._umap_ref_points_data is None:
                return
            if self._umap_world_centroid is None:
                return
            f = value / 100.0
            cy, cx = self._umap_world_centroid
            centroid = np.array([[cy, cx]])
            self.points_layer.data = (
                centroid + (self._umap_ref_points_data - centroid) * f
            )
            if self._umap_channel_layers and self._umap_thumb_cache is not None:
                self._rebuild_umap_canvas()

        def _render_image_umap(self):
            """Downscale crops once, cache, then paste at point positions.

            Subsequent slider changes call ``_rebuild_umap_canvas`` which
            re-pastes cached thumbnails — pure numpy, no PIL, instant.
            """
            if self._view_umap is None or self._view_indices is None:
                return

            self._remove_image_umap()

            valid_mask = ~np.isnan(self._view_umap).any(axis=1)
            valid_positions = np.where(valid_mask)[0]
            if len(valid_positions) < 2:
                return

            crop_entries: List[Tuple[int, int, np.ndarray]] = []
            for i, li_pos in enumerate(valid_positions):
                if li_pos in self._hidden_indices:
                    continue
                meta_idx = self._view_indices[li_pos]
                row = self.metadata.iloc[meta_idx]
                rgb = self._crop_cache.get((row["well_name"], row["experiment"]))
                if rgb is not None:
                    crop_entries.append((i, li_pos, rgb))

            if not crop_entries:
                return

            crop_h, crop_w = crop_entries[0][2].shape[:2]
            aspect = crop_w / max(1, crop_h)

            umap_valid = self._view_umap[valid_positions]

            from scipy.spatial import cKDTree

            tree = cKDTree(umap_valid)
            dists, _ = tree.query(umap_valid, k=2)
            median_nn = float(np.median(dists[:, 1]))
            if median_nn < 1e-8:
                median_nn = 1.0

            thumb_world_h = median_nn

            # Target ~60 px tall, scales down with well count to keep canvas size sane.
            thumb_px_h = max(8, min(80, 4000 // max(1, int(len(crop_entries) ** 0.5))))
            thumb_px_w = max(8, int(thumb_px_h * aspect))

            ppu = thumb_px_h / (thumb_world_h + 1e-8)

            from PIL import Image as PILImage

            thumbs: List[Tuple[int, np.ndarray]] = []
            for _vi, li_pos, rgb in crop_entries:
                thumb = np.array(
                    PILImage.fromarray(rgb).resize(
                        (thumb_px_w, thumb_px_h), PILImage.LANCZOS
                    )
                )
                thumbs.append((li_pos, thumb))

            self._umap_thumb_cache = thumbs
            self._umap_thumb_size = (thumb_px_h, thumb_px_w)
            self._umap_ppu = ppu

            self._rebuild_umap_canvas()

            log.info(
                f"Rendered image UMAP: {len(thumbs)} wells, "
                f"thumb={thumb_px_h}x{thumb_px_w}px"
            )

        def _rebuild_umap_canvas(self):
            """Re-paste cached thumbnails at the points' current positions."""
            if self._umap_thumb_cache is None or self.points_layer is None:
                return

            thumb_px_h, thumb_px_w = self._umap_thumb_size
            ppu = self._umap_ppu
            pts = self.points_layer.data  # (N, 2) — (y, x) in world coords

            positions = []
            for li_pos, _thumb in self._umap_thumb_cache:
                wy, wx = pts[li_pos]
                positions.append((wy, wx))

            if not positions:
                return

            world_ys = np.array([p[0] for p in positions])
            world_xs = np.array([p[1] for p in positions])

            max_canvas_px = 32000
            range_y = world_ys.max() - world_ys.min() + thumb_px_h / max(ppu, 1e-8)
            range_x = world_xs.max() - world_xs.min() + thumb_px_w / max(ppu, 1e-8)
            cur_ppu = min(
                ppu,
                max_canvas_px / (range_y + 1e-8),
                max_canvas_px / (range_x + 1e-8),
            )

            margin = max(thumb_px_h, thumb_px_w) // 2 + 4
            origin_y = world_ys.min() - thumb_px_h / (2 * cur_ppu)
            origin_x = world_xs.min() - thumb_px_w / (2 * cur_ppu)

            px_y = ((world_ys - origin_y) * cur_ppu + margin).astype(int)
            px_x = ((world_xs - origin_x) * cur_ppu + margin).astype(int)

            canvas_h = int(px_y.max()) + thumb_px_h // 2 + margin + 1
            canvas_w = int(px_x.max()) + thumb_px_w // 2 + margin + 1

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for idx, (_li_pos, thumb) in enumerate(self._umap_thumb_cache):
                cy, cx = px_y[idx], px_x[idx]
                tl_y = cy - thumb_px_h // 2
                tl_x = cx - thumb_px_w // 2
                sy = max(0, -tl_y)
                sx = max(0, -tl_x)
                tl_y = max(0, tl_y)
                tl_x = max(0, tl_x)
                br_y = min(canvas_h, tl_y + thumb_px_h - sy)
                br_x = min(canvas_w, tl_x + thumb_px_w - sx)
                sh, sw = br_y - tl_y, br_x - tl_x
                if sh > 0 and sw > 0:
                    canvas[tl_y:br_y, tl_x:br_x] = thumb[sy:sy + sh, sx:sx + sw]

            inv_scale = 1.0 / cur_ppu
            translate_y = origin_y - margin * inv_scale
            translate_x = origin_x - margin * inv_scale

            if self._umap_channel_layers:
                layer = self._umap_channel_layers[0]
                layer.data = canvas
                layer.scale = [inv_scale, inv_scale]
                layer.translate = [translate_y, translate_x]
            else:
                layer = self.viewer.add_image(
                    canvas,
                    name="Fish UMAP",
                    rgb=True,
                    blending="additive",
                    opacity=1.0,
                    scale=[inv_scale, inv_scale],
                    translate=[translate_y, translate_x],
                )
                self._umap_channel_layers.append(layer)

                # Keep points + lasso layers on top.
                for top_layer in [self.points_layer, self.lasso_layer]:
                    if top_layer is not None and top_layer in self.viewer.layers:
                        self.viewer.layers.move(
                            self.viewer.layers.index(top_layer),
                            len(self.viewer.layers) - 1,
                        )
                self.viewer.layers.selection.clear()
                if self._lasso_mode and self.lasso_layer is not None:
                    self.viewer.layers.selection.add(self.lasso_layer)
                elif self.points_layer is not None:
                    self.viewer.layers.selection.add(self.points_layer)

        def _remove_image_umap(self):
            for layer in self._umap_channel_layers:
                try:
                    self.viewer.layers.remove(layer)
                except (ValueError, AttributeError):
                    pass
            self._umap_channel_layers = []
            self._umap_thumb_cache = None

    return LabelTool


def LabelTool(
    viewer,
    prefix: str,
    channels: List[str],
    well_ids: List[str],
    well_names: List[str],
    well_crops: List[Dict[str, np.ndarray]],
    per_channel_embeddings: Dict[str, np.ndarray],
    per_channel_indices: Dict[str, np.ndarray],
    cluster_strategy,
    store: LabelStore,
    parent=None,
):
    """Factory wrapper — defers Qt/napari imports until first call.

    Mirrors ``FindingDory`` in ``fish_sorter.GUI.finding_dory``: importing
    this module is cheap; constructing the widget triggers the heavy
    imports inside ``_build_label_tool``.
    """
    cls = _build_label_tool()
    return cls(
        viewer=viewer,
        prefix=prefix,
        channels=channels,
        well_ids=well_ids,
        well_names=well_names,
        well_crops=well_crops,
        per_channel_embeddings=per_channel_embeddings,
        per_channel_indices=per_channel_indices,
        cluster_strategy=cluster_strategy,
        store=store,
        parent=parent,
    )
