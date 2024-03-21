'''
Assembles mosaics captured via MM

Metadata relies on .pos file
'''
import json
import numpy as np

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
# from time import perf_counter

from skimage import io
    

class MosaicStitcher:
    def __init__(self, parent_dir, metadata_file, prefix):

        self.mosaic = None

        # Save input args
        self.prefix = prefix
        self.parent_dir = Path(parent_dir)

        # Load metadata
        f = open(self.parent_dir / metadata_file)
        data = json.load(f)

        # Get metadata
        self.pos_data = self._extract_pos_metadata(data)
        self.x_overlap, self.y_overlap, self.num_rows, self.num_cols = self._extract_mosaic_params(data)

        # Get image parameters
        dims, self.dtype = self._extract_img_params(data)
        self.img_y_dim, self.img_x_dim, self.img_z_dim = dims

    def _extract_pos_metadata(self, data):
        '''
        Returns a list of metadata dicts.
        
        The list of dicts is indexed by [row][col]
        Each dict includes metadata for the image in that position.
        
        Dict keys:
        - Filename: Absolute path to image
        - Idx: Position within ome.tiff ordering
        '''
        all_pos = data['map']['StagePositions']['array']

        self.x_overlap = int(all_pos[0]['Properties']['scalar']['OverlapPixelsX']['scalar'])
        self.y_overlap = int(all_pos[0]['Properties']['scalar']['OverlapPixelsY']['scalar'])

        self.num_rows = int(max([pos['GridRow']['scalar'] for pos in all_pos])) + 1
        self.num_cols = int(max([pos['GridCol']['scalar'] for pos in all_pos])) + 1

        pos_data = [[None] * self.num_cols for i in range(0, self.num_rows)]
        for i, pos in enumerate(all_pos):
            filename = self.parent_dir / f"{self.prefix}{pos['Label']['scalar']}.ome.tif"
            entry = {
                'Filename': filename,
                # 'OverlapPixelsX': pos['Properties']['scalar']['OverlapPixelsX']['scalar'],
                # 'OverlapPixelsY': pos['Properties']['scalar']['OverlapPixelsY']['scalar'],
                'Idx': i,
            }
            pos_data[pos['GridRow']['scalar']][pos['GridCol']['scalar']] = entry

        return pos_data
    
    def _extract_mosaic_params(self, data):
        '''
        Returns mosaic parameters as (x overlap [pix], y overlap [pix], # of rows, # of cols)
        '''
        all_pos = data['map']['StagePositions']['array']

        return (
            int(all_pos[0]['Properties']['scalar']['OverlapPixelsX']['scalar']),
            int(all_pos[0]['Properties']['scalar']['OverlapPixelsY']['scalar']),
            int(max([pos['GridRow']['scalar'] for pos in all_pos])) + 1,
            int(max([pos['GridCol']['scalar'] for pos in all_pos])) + 1,
        )

    def _extract_img_params(self, data):
        '''
        Returns image parameters as (dim, dtype)

        dim includes (image width [pix], image height [pix], z stack count)
        '''
        im = self._get_img(0, 0)
        return im.shape, im.dtype

    def _get_img(self, row, col):
        '''
        Returns a single image, including all z-stacks
        '''
        filename = self.pos_data[row][col]['Filename']
        all_im = io.imread(filename.resolve())

        return all_im[self.pos_data[row][col]['Idx'], :, :, :]

    def assemble_mosaic(self):
        '''
        Assemble mosaic
        '''
        mosaic_x_dim = (self.img_x_dim * self.num_cols) - (self.x_overlap * (self.num_cols- 1))
        mosaic_y_dim = (self.img_y_dim * self.num_rows) - (self.y_overlap * (self.num_rows - 1))

        mosaic = np.zeros((mosaic_y_dim, mosaic_x_dim, self.img_z_dim), dtype=np.uint32)

        x_translation = self.img_x_dim - self.x_overlap
        y_translation = self.img_y_dim - self.y_overlap

        # Assemble mosaic
        print("Stitching images together")
        for row in tqdm(range(self.num_rows), desc="Row"):
            y_start = row * y_translation
            for col in tqdm(range(0, self.num_cols), desc="Column"):
                x_start = col * x_translation

                mosaic[y_start : y_start + self.img_y_dim, x_start : x_start + self.img_x_dim, :] += self._get_img(row, col)

        # Take average of overlapping areas
        print("Taking average of overlapping areas")
        for row in tqdm(range(1, self.num_rows), desc="Row"):
            y_start = row * y_translation
            mosaic[y_start : y_start - y_translation + self.img_y_dim, :, :] = np.floor_divide(
                mosaic[y_start : y_start - y_translation + self.img_y_dim, :, :],
                2
            ).astype(np.uint32)
        for col in tqdm(range(1, self.num_cols), desc="Column"):
            x_start = col * x_translation
            mosaic[:, x_start : x_start - x_translation + self.img_x_dim, :] = np.floor_divide(
                mosaic[:, x_start : x_start - x_translation + self.img_x_dim, :],
                2
            ).astype(np.uint32)

        self.mosaic = mosaic.astype(self.dtype)

    def get_mosaic(self):
        '''
        Returns assembled mosaic
        '''
        if self.mosaic is None:
            raise ValueError("No mosaic found! Run assemble_mosaic() first.")
        return self.mosaic

    def display_mosaic(self, zslice):
        '''
        Displays specified z-stack of assembled mosaic
        '''
        if zslice >= self.img_z_dim:
            raise ValueError(f"Selected z-slice does not exist. Select a z-slice within the range 0-{self.img_z_dim-1}.")

        io.imshow(mosaic[:, :, zslice])
        io.show()

    def save_mosaic(self, dir=None, prefix=None):
        '''
        Saves each z-stack of the mosaic as a separate .tiff

        By default, saves images according to parent_dir and prefix specified in __init__
        Overwrite default dir and prefix with optional arguments
        '''
        # Set default vals if empty
        if dir is None:
            dir = self.parent_dir
        if prefix is None:
            prefix = self.prefix

        dir = Path(dir)
        if not dir.exists():
            raise FileNotFoundError(f"Directory {dir} does not exist.")

        print("Saving file(s) to {dir}")
        if self.mosaic is None:
            raise ValueError("No mosaic found! Run assemble_mosaic() first.")

        # Save each z-stack separately, to prevent files from getting too large
        for i in range(self.img_z_dim):
            filename = dir / f'{prefix}mosaic{i}.tiff'
            io.imsave(filename, self.mosaic[:, :, i])
            print(f'Saved z-slice {i} to {filename}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--dir',
        required=True,
        help='Parent directory containing files',
    )
    parser.add_argument(
        '-m', '--metadata',
        required=True,
        help='Metadata filename, defined relative to parent directory',
    )
    parser.add_argument(
        '-p', '--prefix',
        required=True,
        help='Image file prefix, excluding micromanager label. '
            '(Eg. the prefix for A3_5x_10x10_MMStack_7-Pos000_001.ome.tif is A3_5x_10x10_MMStack_)'
    )

    args = parser.parse_args()
    stitcher = MosaicStitcher(args.dir, args.metadata, args.prefix)

    stitcher.assemble_mosaic()
    stitcher.save_mosaic()