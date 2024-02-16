import pymmcore

from pymmcore_plus import CMMCorePlus
from pathlib import Path

parent_dir = Path("C:/Program Files/Micro-Manager-2.0-20240130")
config_file = "test.cfg"
config_dir = parent_dir / config_file

mmc = CMMCorePlus()
mmc.setDeviceAdapterSearchPaths([str(parent_dir)])
mmc.loadSystemConfiguration("test.cfg")

mmc.snapImage()
print(mmc.getImage())
