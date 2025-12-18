# python-fish-sorter

This repository contains the Python application and custon __python-fish-sorter__ package for control of the Fish Sorter v2 using pymmcore+. Includes image acquisition and processing tools, such as mosaic capture/stitching. 

The Fish Sorter is a custom-built system for automated screening and selection of zebrafish embryos and larvae with desired features of interest. The system is built around a commercial inverted epi-fluorescent microscope for fish imaging, plus additional custom hardware for fish retrieval. 

The fish sorter was developed by the CZ Biohub SF Bioengineering and Jacobo groups.

Maintenance of this repo is the responsibility of Diane Wiener. Plese direct any communication via creation of an Issue at the project repo [here](https://github.com/czbiohub-sf/python-fish-sorter/issues).

This source describes Open Hardware, which is licensed under the CERN-OHL-W v2. Software is licensed under BSD 3-Clause.

Copyright 2025, Chan Zuckerberg Biohub San Francisco.

## Installation with uv

This package uses [uv](https://docs.astral.sh/uv/?utm_source=chatgpt.com) to manage reproducible installs and package dependencies.

The ground truth package lock is based on the software installed for the fish sorter v2 instrument running Windows with Python 3.12. The pinned versions are frozen in pyproject.toml + uv.lock.

1. Prerequisites

  * Python 3.12 installed on your system.

  * [uv](https://docs.astral.sh/uv/?utm_source=chatgpt.com)

    * Mac/Linux
      ```
        curl -LsSf https://astral.sh/uv/install.sh | sh
      ```
    * Add to PATH if needed
      ```
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zprofile
      ```

    * Windows (PowerShell)
      ```
      winget install --id=astral-sh.uv -e
      ```
    * Restart PowerShell or add $HOME\.local\bin to PATH

2. Clone the repository

    Clone into your preferred directory (e.g. ~/Documents/GitHub):

    * Mac/Linux
      ```
      cd ~/Documents/GitHub
      git clone https://github.com/<your-org-or-username>/python-fish-sorter.git
      cd python-fish-sorter
      ```

    * Windows (PowerShell)
      ```
      cd $env:USERPROFILE\Documents\GitHub
      git clone https://github.com/<your-org-or-username>/python-fish-sorter.git
      cd python-fish-sorter
      ```
3. First-time setup
   ```
   uv sync
   ```
   
4. Running the project

    Inside the repo:
    ```
    uv run python fish_sorter/GUI/fish_picker.py
    ```

5. Running one-off scripts outside the project

    If you keep helper scripts in a separate repo or folder (e.g. ~/Documents/GitHub/scripts), you can still run them using the fish_sorter environment:
    ```
    cd ~/Documents/GitHub/scripts
    uv --project ../python-fish-sorter run python my_script.py
    ```

    or with an absolute path:
    ```
    uv --project ~/Documents/GitHub/python-fish-sorter run python ~/Documents/GitHub/scripts/my_script.py
    ```

6. Updating dependencies

    Add a new package:
      ```
      uv add somepackage==1.2.3
      uv lock
      ```

    Commit both ```pyproject.toml``` and ```uv.lock```.

    Re-sync environment after changes:
    ```
    uv sync
    ```

7. Notes

    * Always use Python 3.12.

    * The fish sorter v2 instrument environment is the “ground truth.” If you change dependencies, re-lock on instrument first ```uv lock```.

## Local configuration
Follow setup for [napari-micromanager](https://github.com/pymmcore-plus/napari-micromanager) to configure to the specific microscopre hardware:
* Use `fish_sorter/GUI/nmm.py` to test run napari-micromanager microscope control
* Use `fish_sorter/GUI/nmm_basic.py` to test run napari-micromanager microscope control with basic ancillary hardware control 

In case of hardware or software installation differences, the following config files may need updating:
* In case of different Zaber stages, update the stage names in `fish_sorter/configs/hardware/zaber_config.json`
* In case of different micromanager configuration, update `MM_DIR` in `fish_sorter/paths.py`

NOTE: The stage control class in `zaber_controller.py` is intended to work with Zaber's binary library for compatibility with older stages. Newer stages may have compatibility issues with the binary library and require Zaber's ASCII library instead.
