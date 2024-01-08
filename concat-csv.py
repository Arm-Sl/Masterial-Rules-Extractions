import pandas as pd
import argparse


def concat_csv(Mpath, Cpath, dirSave):
    main = pd.read_csv(Mpath,sep=" ")
    labels = pd.read_csv(Cpath, sep=" ")
    df = pd.concat([main, labels], axis = 1)
    df.to_csv(dirSave, sep=",", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="concat csv", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--main", type=str, default=None)
    parser.add_argument("--labels" , type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    concat_csv(args.main, args.labels, args.save_dir)