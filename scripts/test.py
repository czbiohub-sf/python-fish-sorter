import pymmcore
import argparse
import numpy as np
import useq

from pymmcore_plus import CMMCorePlus
from pathlib import Path

# TEMP
import matplotlib.pyplot as plt


parent_dir = Path("C:/Program Files/Micro-Manager-2.0-20240130")
config_file = "test.cfg"
config_dir = parent_dir / config_file

class Imager():

	def __init__(self, mm_dir, save_dir, cfg_file=None):

		self.mmc = CMMCorePlus()
		self.mmc.setDeviceAdapterSearchPaths([mm_dir])

		self.save_dir = Path(save_dir)

		# TEMP
		self.i = 0

		# Configs
		self.channels = None
		self.stage_positions = None
		self.grid_plan = None
		self.z_plan = None
		self.axis_order = "cpgz" # ie. at each g, do a full z iteration
		
		if cfg_file is None:
			# Load demo config by default
			mmc.loadSystemConfiguration()
		else:
			mmc.loadSystemConfiguration(cfg_file)

		mmc.snapImage()
		print(mmc.getImage())

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
		mda_sequence = MDASequence(
			channels=self.channels,
			stage_positions=self.stage_positions,
			grid_plan=self.grid_plan,
			z_plan=self.z_plan,
			axis_order=self.axis_order,  
		)
		self.save_sequence('test.yaml', mda_sequence)

		# Run it!
		self.mmc.run_mda(mda_sequence)

	def pause(self);
		self.mmc.mda.toggle_pause()

	def cancel(self);
		self.mmc.mda.cancel()

	def save_sequence(self, file, mda_sequence):
		(self.save_dir / file).write_text(mda_sequence.yaml())

	@mmc.mda.events.frameReady.connect 
	def on_frame(image: np.ndarray, event: useq.MDAEvent):
		print(
			f"received frame: {image.shape}, {image.dtype} "
			f"@ index {event.index}, z={event.z_pos}"
		)

		# Save image here
		# How to save multichannel tiff?? 
		# TEMP
		plt.imsave(self.save_dir / f"{i}.png", image)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mmdir')
	parser.add_argument('--cfg')
	args = parser.parse_args()

	Imager(args.mm_dir, args.cfg)
