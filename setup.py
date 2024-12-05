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
<<<<<<< HEAD:fish_sorter/setup.py
        'useq',
        'abc',
=======
        'useq-schema',
>>>>>>> classification:setup.py
        'argparse',
        'numpy',
        'pymodbus',
        'pymmcore-widgets',
    ],
)