import subprocess


def main():
    DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes/*/*"
    OUTPUT_PATH = "/home/satyarth934/data/nasa_impact/hurricanes_reorganized"

    img_paths = glob.glob(DATA_PATH)
    print("len(img_paths):", len(img_paths))
    random.shuffle(img_paths)

    train_test_split = 0.8
    X_train_paths = img_paths[:int(train_test_split * len(img_paths))]
    X_test_paths = img_paths[int(train_test_split * len(img_paths)):]

    subprocess.call("mkdir -p " + OUTPUT_PATH + "/train", shell=True)
    for xtr in tqdm.tqdm(X_train_paths):
        subprocess.call("cp %s %s" % (xtr, OUTPUT_PATH + "/train/"), shell=True)

    subprocess.call("mkdir -p " + OUTPUT_PATH + "/test", shell=True)
    for xte in tqdm.tqdm(X_test_paths):
        subprocess.call("cp %s %s" % (xte, OUTPUT_PATH + "/test/"), shell=True)


if __name__ == "__main__":
    main()