import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get label for normal video')
    parser.add_argument('json1', type=str, help='json 1')
    parser.add_argument('json2', type=str, help='json 2')
    parser.add_argument('out_json', type=str, help='out json')
    args = parser.parse_args()

    return args

args = parse_args()
if __name__ == '__main__':
    with open(args.json1, 'r') as f:
        json1 = json.load(f)

    with open(args.json2, 'r') as f:
        json2 = json.load(f)

    json1.update(json2)

    with open(args.out_json, 'w') as f:
        json.dump(json1, f)


