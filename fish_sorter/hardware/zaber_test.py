import sys
from json import load
from pathlib import Path
from time import sleep
from typing import List, Optional

from zaber_controller import ZaberController

def main() -> bool:
    """
    Test zaber stage function

    :raises Exception: logs if the hardware config file not found or hardware does not connect
    :return: A bool as to whether a the code ran successfully through all steps.
    :rtype: bool
    """

    cfg_dir = Path().absolute().parent / "configs/hardware"
    cfg_file = "zaber_config.json"
    cfg_path = cfg_dir / cfg_file
    # Initialize and connect to hardware controller
    try:
        with open(cfg_path, 'r') as f:
                p = load(f)
        zaber_config = p['zaber_config']
        zc = ZaberController(zaber_config, env='prod')

        pick_cfg_dir = Path().absolute().parent / "configs/"
        pick_cfg_file = "picker_defaults_config.json"
        pick_cfg_path = pick_cfg_dir / pick_cfg_file
        with open(pick_cfg_path, 'r') as f:
            p = load(f)
            pick_config = p['defaults']

        proceed = True
    except Exception as e:
        print("Could not initialize and connect hardware controller")
        proceed = False
    
    if proceed:
        # Test moving the pipette, x, and y stages to max position
        stages = ['x', 'y', 'p']

        print('Move stages to max and back home')
        for stage in stages:
            zc.move_arm(stage, zaber_config['max_position'][stage])
            sleep(2)
            zc.move_arm(stage, zaber_config['home'][stage])
            sleep(2)

        print('Move pipette to set locations')
        print('Swing height')
        zc.move_arm('p', zaber_config['pipette_swing']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)
        
        print('Pick height')
        zc.move_arm('p', pick_config['pipette']['stage']['pick']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Clearance height')
        zc.move_arm('p', pick_config['pipette']['stage']['clearance']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Dispense height')
        zc.move_arm('p', pick_config['pipette']['stage']['dispense']['p'])
        sleep(2)
        zc.move_arm('p', zaber_config['home']['p'])
        sleep(2)

        print('Test complete')
        zc.disconnect()

    return proceed
        

if __name__ == "__main__":
    proceed = main()

    if not proceed:
        print("Exited with error(s)")
    else:
        print("Exited with no errors")