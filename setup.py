from setuptools import setup, find_packages

setup(
    name="fish_sorter",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        'pyqt5',
        'napari',
        'napari-micromanager',
        'zaber_motion',
        'tqdm',
        'iter-tools',
        'useq-schema',
        'argparse',
        'numpy',
        'pymodbus',
        'pymmcore-widgets',
    ],
)