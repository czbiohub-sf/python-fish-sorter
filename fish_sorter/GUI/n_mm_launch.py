import napari
import napari_micromanager
import numpy as np
import os
import pymmcore_plus

from pathlib import Path
from useq import MDASequence, Position
from GUI.pipette_gui import PipetteWidget

os.environ['MICROMANAGER_PATH'] = "C:/Program Files/Micro-Manager-2.0-20240130"
micromanager_path = os.environ.get('MICROMANAGER_PATH')

def nmm():

    cfg_dir = Path().absolute().parent / "fish_sorter/configs/micromanager"
    cfg_file = "20240718 - LeicaDMI - AndorZyla.cfg"
    cfg_path = cfg_dir / cfg_file
    print(cfg_path)

    v = napari.Viewer()
    dw, main_window = v.window.add_plugin_dock_widget("napari-micromanager")
    
    core = main_window._mmc
    core.loadSystemConfiguration(str(cfg_path))

    sequence = MDASequence(
        channels = [
            {"config": "GFP","exposure": 100}, 
            {"config": "TXR", "exposure": 100}
        ],
        stage_positions = [
            {"x": 110495.44, "y": 10863.76, "z": 2779.09, "name": "top_R"},
            {"x": 17883.77, "y" : 10166.54, "z": 2779.09, "name": "top_L"},
            {"x": 110495.44, "y": 73208.59, "z": 2776.70, "name": "bot_R"},
            {"x": 17492.82, "y": 73208.58, "z": 2776.70, "name": "bot_L"},
            Position(
                x=17883.77, y=10166.54, z=2779.09, name= "array", 
                sequence=MDASequence(
                    grid_plan={"rows": 13, "columns": 18, "relative_to": "top_left", "overlap": 5, "mode": "row_wise_snake"})
            ),
        ],
        axis_order = "pc",
    )

    main_window._show_dock_widget("MDA")
    mda_widget = v.window._dock_widgets.get("MDA").widget()
    mda_widget.setValue(sequence)

    v.window._qt_viewer.console.push(
        {"main_window": main_window, "mmc": core, "sequence": sequence, "np": np}
    )

    pipette = PipetteWidget()
    v.window.add_dock_widget(pipette, name='pipette')

    napari.run()

if __name__ == "__main__":
    nmm()