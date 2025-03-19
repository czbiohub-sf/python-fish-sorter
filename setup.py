from setuptools import setup, find_packages

setup(
    name="fish_sorter",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        'pyqt6',
        'napari',
        'napari-micromanager',
        'zaber_motion',
        'tqdm',
        'iter-tools',
        'useq-schema',
        'argparse',
        'matplotlib',
        'numpy',
        'pymodbus',
        'pymmcore-widgets',
    ],
)