import pandas as pd
import argparse
import json


def concat_csv(Mpath, Cpath, dirSave):
    main = pd.read_csv(Mpath,sep=" ")
    labels = pd.read_csv(Cpath, sep=" ")
    df = pd.concat([main, labels], axis = 1)
    df.to_csv(dirSave, sep=",", index=False)

def concat_json(Lpath, Apath, dirsave):
    with open(dirsave, "w") as save:
        Lore = json.load(open(Lpath,"r"))
        Anchor = json.load(open(Apath,"r"))
        concat = Lore + Anchor
        json.dump(concat, save, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="concat csv", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--first", type=str, default=None)
    parser.add_argument("--second" , type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    match args.type:
        case "csv":
            concat_csv(args.first, args.second, args.save_dir)
        case "json":
            concat_json(args.first, args.second, args.save_dir)