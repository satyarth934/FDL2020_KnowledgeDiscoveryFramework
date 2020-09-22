import sys
sys.dont_write_bytecode = True
import glob
import numpy as np

import utils


DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes/*/*"


def main():
    dims = (448, 448, 3)
    
    # Dataloader creation and test
    img_paths = glob.glob(DATA_PATH)
    print("len(img_paths):", len(img_paths))
    img_paths = utils.getUsableImagePaths(image_paths=img_paths, data_type="png")
    print("---------")
    print("len(usable img_paths):", len(img_paths))
    f = open("hurricane_input.txt", 'w')
    f.writelines("\n".join(img_paths))
    f.close()


if __name__ == "__main__":
    main()