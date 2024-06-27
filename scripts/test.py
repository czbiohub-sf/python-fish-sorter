import pymmcore
import argparse
import numpy as np
import useq

from pymmcore_plus import CMMCorePlus
from pathlib import Path
from skimage import io
from time import perf_counter


save_dir = Path("C:/Users/Chan Zuckerberg/Documents/data_mk")
cfg_dir = Path("C:/Users/Chan Zuckerberg/Documents/python-fish-sorter/micromanager-configs")
cfg_file = "20240222 - LeicaDMI - AndorZyla.cfg"
mm_dir = Path("C:/Program Files/Micro-Manager-2.0-20240130")
cfg_path = cfg_dir / cfg_file

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

    def set_channels(self):
        # Determine a good way to make this configurable
        self.channels = [
            {"config": "DAPI", "exposure": 50},
            {"config": "FITC", "exposure": 80},
        ]

    def set_pos0(self, x0, y0, z0):
        self.stage_positions = [(x0, y0, z0)]

    def set_grid(self, width, height, rows, cols):
        self.grid_plan = {"fov_width": width, "fov_height": height, "rows": rows, "columns": cols}

    def set_zstack(self, range, step):
        self.z_plan = {"range": range, "step": step}

    def image(self):
        # mda_sequence = useq.MDASequence(
        #     channels=self.channels,
        #     stage_positions=self.stage_positions,
        #     grid_plan=self.grid_plan,
        #     z_plan=self.z_plan,
        #     axis_order=self.axis_order,  
        # )
        mda_sequence = useq.MDASequence(
            channels=self.channels,
            stage_positions=self.stage_positions,
            grid_plan=self.grid_plan,
            z_plan=self.z_plan,
            axis_order=self.axis_order,  
        )        
        self.save_sequence('test.yaml', mda_sequence)

        # Run it!
        self.mmc.run_mda(mda_sequence)

    def pause(self):
        self.mmc.mda.toggle_pause()

    def cancel(self):
        self.mmc.mda.cancel()

    def save_sequence(self, file, mda_sequence):
        (self.save_dir / file).write_text(mda_sequence.yaml())

    def dec(func):
        def hidden(self):
            self.mmc.mda.events.frameReady.connect(func)

    @dec 
    def on_frame(self, image: np.ndarray, event: useq.MDAEvent):
        print(
            f"received frame: {image.shape}, {image.dtype} "
            f"@ index {event.index}, z={event.z_pos}"
        )

        # Save image here
        io.imsave(self.save_dir / f"{self.prefix}_{vent.index}.tif", image)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mmdir')
    # parser.add_argument('--cfg')
    # args = parser.parse_args()

    # Imager(args.mm_dir, args.cfg)
    Imager(mm_dir, save_dir, cfg_path=cfg_path, prefix="test")
