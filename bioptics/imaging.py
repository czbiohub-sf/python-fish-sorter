import json
import numpy as np

# f = open('/Volumes/KINGSTON/test_1/test_MMStack_2-Pos000_000_metadata.txt')
# data = json.load(f)

# def extract_mosaic_metadata(data):
    
class Imaging:
    def __init__(self, filepath='/Volumes/KINGSTON/test_1/test_MMStack_2-Pos000_000_metadata.txt'):
        # filepath = '/Volumes/KINGSTON/test_1/test_MMStack_2-Pos000_000_metadata.txt'

        f = open(filepath)
        data = json.load(f)
        self.pos_data = self._extract_img_metadata(data)

    def _extract_img_metadata(self, data):
        '''
        Returns a list indexed by [row][col]
        Each entry includes [filename, x pos (um), y pos (um)]
        '''
        pos_json = data['Summary']['StagePositions']
        
        num_rows = max([pos['GridRow'] for pos in pos_json]) + 1
        num_cols = max([pos['GridCol'] for pos in pos_json]) + 1

        pos_data = num_rows * [num_cols * [None]]

        for pos in pos_json:
            entry = [pos['Label']] + pos['DevicePositions'][0]['Position_um']
            pos_data[pos['GridRow']][pos['GridCol']] = entry

        return pos_data

    def get_img_name(self, row, col):
        return self.pos_data[row][col][0]

    def get_img_pos(self, row, col):
        return self.pos_data[row][col][1], self.pos_data[row][col][2]

if __name__ == '__main__':
    a = Imaging()
    print(a.get_img_name(10, 0))
    print(a.get_img_pos(10, 0))