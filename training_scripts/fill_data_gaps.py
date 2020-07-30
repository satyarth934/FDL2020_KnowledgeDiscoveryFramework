# 45 pixels: radius of the locality
# plt.imshow(image)
# plt.show()

import random
import subprocess
from multiprocessing import Pool


def getClosest(pt, options):
    x,y = pt
    min_val = -9999
    min_pt = options[0]
    for o in options:
        sse = np.sum(np.square(np.array(pt) - np.array(o)))
        if sse < min_val:
            min_val = se
            min_pt = o
    
    return min_pt


def fillGaps(DATA_PATH, OUTPUT_PATH):
    img_paths = glob.glob(DATA_PATH)
    print(len(img_paths))

    dims = (448, 448, 3)

    # Loading Data
    dataset = utils.getData(img_paths, dims)
    
    N = 2
    radius = 85

    for path_idx, image in enumerate(tqdm.tqdm(dataset)):
        image = utils.normalize(image)
        
        # local region
        (x,y,z) = np.where(np.isnan(image))

        # nan value indices
        nan_idxs = [_ for _ in zip(x,y)]
        nan_idxs = list(set(nan_idxs))

        attempts = 0

        clone = np.copy(image)

        for i, item in enumerate(nan_idxs):
            x,y = item
            # region
            xtl, xtr = max(0, x - radius), min(image.shape[1], x + radius)
            ytl, ytr = max(0, y - radius), min(image.shape[1], y + radius)

            random_idxs = []
            counter = 0
            att2 = 0
            flag = 0
            while counter < N:
                if att2 > N * 10:
                    flag = 1
                    break
                att2 += 1

                attempts += 1
                random_idx = np.random.randint(xtl, xtr), np.random.randint(ytl, ytr)
                if np.sum(np.isnan(image[random_idx[0], random_idx[1],:])) > 0:
                    continue
                random_idxs.append(random_idx)
                counter += 1

            if i%5000 == 0:
                print("attempts:", attempts / ((i+1) * N))

            if(flag==1):
                continue

            closest_x, closest_y = getClosest((x,y), random_idxs)
            clone[x,y,:] = image[closest_x,closest_y,:]

        np.save(OUTPUT_PATH + "/" + os.path.basename(img_paths[path_idx]), clone)
#         plt.imshow(clone)
        # plt.savefig("n%drad%dA3.png" % (N, radius))
#         plt.show()


def main():
    DATA_PATH = "/home/fcivilini/modis_output_terra/array_3bands_adapted/448/median_removed"
    OUTPUT_PATH = "/home/satyarth934/modis_output_terra/array_3bands_adapted/448/median_removed_gap_filled"
    subprocess.call(("mkdir -p " + OUTPUT_PATH), shell=True)    
    
    fillGaps(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()