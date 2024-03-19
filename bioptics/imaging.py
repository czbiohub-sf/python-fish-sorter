import json
import numpy as np

from pathlib import Path

from PIL import Image
from skimage import io
from matplotlib import pyplot as plt

# f = open('/Volumes/KINGSTON/test_1/test_MMStack_2-Pos000_000_metadata.txt')
# data = json.load(f)

# def extract_mosaic_metadata(data):


    
class Imaging:
    def __init__(
        self,
        parent_dir='/Users/michelle.khoo/Desktop/A3_5x_10x10_1',
        metadata_file='A3_5x_position_list.pos',
        prefix='A3_5x_10x10_MMStack_',
        ):

        self.prefix = prefix
        self.parent_dir = Path(parent_dir)

        metadata_filepath = self.parent_dir / metadata_file

        f = open(metadata_filepath)
        data = json.load(f)
        # print(data['encod÷ing'])
        # print("HI")

        self.pos_data = self._extract_pos_metadata(data)
        # self.img_dims = self._extract_img_dims(data)

    def _extract_pos_metadata(self, data):
        '''
        Returns a list indexed by [row][col]
        Each entry includes [filename, x pos (um), y pos (um)]
        '''
        pos_json = data['map']['StagePositions']['array']

        # pos_json = data['map']['StagePositions']
        # pos_json = data['StagePositions']

        # for pos in pos_json:
        #     print(pos)
        #     print('\n\n')

        num_rows = max([pos['GridRow']['scalar'] for pos in pos_json]) + 1
        num_cols = max([pos['GridCol']['scalar'] for pos in pos_json]) + 1

        pos_data = [[None] * num_cols for i in range(0, num_rows)]
        for i, pos in enumerate(pos_json):
            filename = self.parent_dir / f"{self.prefix}{pos['Label']['scalar']}.ome.tif"
            entry = {
                'Filename': filename,
                'OverlapPixelsX': pos['Properties']['scalar']['OverlapPixelsX']['scalar'],
                'OverlapPixelsY': pos['Properties']['scalar']['OverlapPixelsY']['scalar'],
                'Idx': i,
            }
            pos_data[pos['GridRow']['scalar']][pos['GridCol']['scalar']] = entry

        return pos_data

    # def

    def get_img(self, row, col):
        filename = self.pos_data[row][col]['Filename']
        im = io.imread(filename.resolve())
        # im = Image.open(filename.resolve())
        # for key in im.tag.keys():
        #     print(im.tag[key])
        # print(im.tag[256])
        # print(np.shape(im))
        # print(np.array(im.seek(0)) - np.array(im.seek(1)))
        # print(im.shape)
        # # print(im)
        # print(filename)
        io.imshow(im[self.pos_data[row][col]['Idx'], :, :, 0])
        # print(im.tag[0, :, :, 0])
        io.show()
        # im[0].show()

        # TODO convert to np?
        return im

    # def stitch_imgs(self):
    #     for img in 

    
    # def _extract_img_dims(self, data):
    #     '''
    #     Returns (width, height) in px
    #     '''
    #     return data['Summary']['Width'], data['Summary']['Height']

    # def get_img_name(self, row, col):
    #     return self.pos_data[row][col][0]

    # def get_img_pos(self, row, col):
    #     return self.pos_data[row][col][1], self.pos_data[row][col][2]

    # def assemble_mosaic():
    #     pass

if __name__ == '__main__':
    a = Imaging()
    # print(a.pos_data)
    a.get_img(3,2)
    # print(a.get_img_name(10, 0))
    # print(a.get_img_pos(10, 0))
    # print(a.pos_data)