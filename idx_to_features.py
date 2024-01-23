import json
import argparse

def idx_to_feature(rules, features, save_dir):
    with open(rules) as json_file:
        with open(features) as feature_names:
            res = list()
            data = json.load(json_file)
            names = json.load(feature_names)
            names = names["feature_names"]
            for rule in data:
                t = dict()
                for idx in rule:
                    if(idx!="label"):
                        t[names[int(idx)]] = rule[idx]
                    else:
                        t[idx] = rule[idx]
                res.append(t)
            with open(save_dir, 'w') as f:
                json.dump(res, f, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="idx to features", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rules", type=str, default=None)
    parser.add_argument("--features" , type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    idx_to_feature(args.rules, args.features, args.save_dir)