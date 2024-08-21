from setuptools import setup, find_packages

setup(
    name="fish_sorter",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'pyqt5',
        'napari',
        'napari-micromanager==0.1.0',
        'zaber_motion',
        'tqdm',
        'iter-tools',
        'useq',
        'argparse',
        'numpy',
        'pymodbus',
    ],
)