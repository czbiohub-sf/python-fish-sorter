import pymmcore
import argparse

from pymmcore_plus import CMMCorePlus
from pathlib import Path


parent_dir = Path("C:/Program Files/Micro-Manager-2.0-20240130")
config_file = "test.cfg"
config_dir = parent_dir / config_file

def run(mm_dir, cfg_file=None):

	mmc = CMMCorePlus()
	mmc.setDeviceAdapterSearchPaths([mm_dir])
	
	if cfg_file is None:
		# Load demo config by default
		mmc.loadSystemConfiguration()
	else:
		mmc.loadSystemConfiguration(cfg_file)

	mmc.snapImage()
	print(mmc.getImage())

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('mm_dir')
	parser.add_argument('--cfg')
	args = parser.parse_args()

	run(args.mm_dir, args.cfg)
