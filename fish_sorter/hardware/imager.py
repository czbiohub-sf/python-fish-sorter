import pymmcore
import argparse
import numpy as np
import useq

from pymmcore_plus import CMMCorePlus
from pathlib import Path
from skimage import io
from time import perf_counter
from typing import Dict, List

from fish_sorter.paths import MM_DIR, SAVE_DIR


class Imager():

    def __init__(self, mm_dir, save_dir, cfg_path=None, prefix=""):

        self.mmc = CMMCorePlus()
        self.mmc.setDeviceAdapterSearchPaths([str(mm_dir)])

        self.save_dir = Path(save_dir)
        self.prefix = prefix

        # Configs
        self.channels = None
        self.stage_positions = None
        self.grid_plan = None
        self.z_plan = None
        self.axis_order = "cpgz" # ie. at each g, do a full z iteration
        
        print("Loading mm config")
        t0 = perf_counter()
        if cfg_path is None:
            # Load demo config by default
            self.mmc.loadSystemConfiguration()
        else:
            self.mmc.loadSystemConfiguration(cfg_path)
        print(f"Finished loading mm config in {perf_counter()-t0} s")

        self.mmc.snapImage()
        print(self.mmc.getImage())
        print("Snapped image")

        print("Capturing mosaic")
        self.image()
        print("Finished capturing mosaic")

    def image_here(self):
        self.mmc.snapImage()
        return self.mmc.getImage()

    def image_mosaic(
        self,
        channels: Dict[str],
        corners_px: List[float], # [TL x, TL y, BR x, BR y]
        fov_dims_px: tuple,
        overlap_frac: float,
        axis_order: str = "cpgz", # ie. at each g, do a full z iteration
    ):
        self.channels = [{"config": ch.key, "exposure": ch.value} for ch in channels]
        mda_sequence = useq.MDASequence(
            channels=[{"config": ch.key, "exposure": ch.value} for ch in channels],
            stage_positions=generatePos(corners_px, fov_dims_px, overlap_frac),
            axis_order=self.axis_order,  
        )        

        # Run it!
        self.mmc.run_mda(mda_sequence)

    def calc_positions(
        self,
        corners_px: np.ndarray, # [ [TL x, TL y], [BR x, BR y] ]
        fov_dims_px: tuple,
        overlap_frac: float,
    ):
    # TODO include z pos?

        def calc_grid_size(total_px, fov_dim, overlap_frac):
            overlap_px = fov_dim * overlap_frac
            return np.ceiling((total_px - overlap_px) / (fov_dim - overlap_px))

        # TODO clean up with numpy style operation
        grid_dims = (
            calc_grid_size(corners_px[0, 1] - corners_px[0, 0], fov_dims_px[0], overlap_frac),
            calc_grid_size(corners_px[1, 1] - corners_px[1, 0], fov_dims_px[1], overlap_frac),
        )

        def calc_pos0(grid_dim, center_px, fov_dim, overlap_frac):
            overlap_px = fov_dim * overlap_frac
            total_px = grid_dim * (fov_dim - overlap_px) - overlap_px
            return center_px - (total_px / 2)

        pos0 = (
            calc_pos0(grid_dims[0], (corners_px[0, 1] + corners_px[0, 0]) / 2, fov_dims_px[0], overlap_frac),
            calc_pos0(grid_dims[1], (corners_px[1, 1] + corners_px[1, 0]) / 2, fov_dims_px[1], overlap_frac),
        )

        return [
            (
                pos0[0] + (fov_dims_px[0] * (1- overlap_frac) * x_index),
                pos0[1] + (fov_dims_px[1] * (1- overlap_frac) * y_index),
            ) for x_index in range(0, grid_dims[0]) for y_index in range(0, grid_dims[1])
        ]

    def pause(self):
        self.mmc.mda.toggle_pause()

    def cancel(self):
        self.mmc.mda.cancel()

    def save_sequence(self, file, mda_sequence):
        (self.save_dir / file).write_text(mda_sequence.yaml())

    def frame_handler(func):
        def hidden(self):
            self.mmc.mda.events.frameReady.connect(func)

    @frame_handler 
    def on_frame(self, image: np.ndarray, event: useq.MDAEvent):
        # print(
        #     f"received frame: {image.shape}, {image.dtype} "
        #     f"@ index {event.index}, z={event.z_pos}"
        # )

        print(
            f"received frame: {image.shape}, {image.dtype} @ index {event.index}, z={event.z_pos}"
        )

        # Save image here
        io.imsave(self.save_dir / f"{self.prefix}_{event.index}.tif", image)


if __name__ == "__main__":
    imager = Imager(MM_DIR, SAVE_DIR, prefix="test")
    imager.imageMosaic()
