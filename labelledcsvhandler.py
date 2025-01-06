import  os, datetime

import numpy as np
import tifffile
# *** Requires RPE_Segmentation
from imagetools import export_mask_id_3d


def usage():
    txt = '''Usage: python csv_to_ids.py <csvfile>.csv [<mask>.tif|<stack>.ome.tif|<RPEMeta>.rpe.json]
    Output (if successful): <csvfile>_ids.tif
    
    If the second parameter is omitted, the script will try to find corresponding tif (either source stack or mask).
    This is only needed to determine the exact shape (n_frames, height, width) of the output .tif stack.
'''
    print(txt)


def error(txt):
    print(txt)
    print()
    usage()

def csvtoids(argv):
    """

    :param argv:
    :return:
    """
    if len(argv) < 2:
        usage()
        # sys.exit(0)

    csvpath = os.path.abspath(argv[1])
    if not os.path.isfile(csvpath):
        error(f'ERROR: No such file: {csvpath}')

    print(f'Input segmentation CSV: {csvpath}')
    basedir, fn = os.path.split(csvpath)
    bn, ext = os.path.splitext(fn)
    tifpath = os.path.join(basedir, bn + '_ids.tif')

    reftif = None
    if reftif is None:
        # Try source tif or .rpe.json

        # Chop off _DNA_RPE or _Actin_RPE from the end of the basename
        for ch in ('_DNA', '_Actin'):
            idx = bn.find(ch)
            if idx >= 0:
                bn = bn[:idx]
                break
        # Go one directory up to look for the source file
        srcdir = os.path.dirname(basedir)
    shape = argv[2]
    print(shape)
    mdata = np.empty(shape=shape, dtype=np.uint16)
    id = export_mask_id_3d(mdata, csvpath)

    if id < 1:
        print(f'ERROR: No objects imported from {csvpath}. Wrong CSV file maybe?')
        print('No output generated.')
    else:
        print()
        print(f'Write output segmentation (TIFF-16) to: {tifpath}')
        tifffile.imwrite(tifpath, mdata, photometric='minisblack', compression='zlib')

    print('OK.')

if __name__ == '__main__':
    csvname = r'C:\rpemrcnn\Results\final_segmentations\FBL\Cell\csv\P1-W2-FBL_G03_F002_DNA_RPE.csv'
    shape = (27,1078, 1278)
    args = [None, csvname, shape]

    csvtoids(args)