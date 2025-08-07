#!/usr/bin/env python3
import argparse
import os

def main():
    p = argparse.ArgumentParser(
        description='Merge newline-separated solution CSVs into one combined solution CSV'
    )
    p.add_argument('input_dir', help='Directory containing solution CSVs')
    p.add_argument('output_csv', help='Path to write the merged CSV')
    args = p.parse_args()

    def sizeof(f):
        return os.path.getsize(os.path.join(args.input_dir, f))
    
    files = sorted( (sizeof(f), f) for f in os.listdir(args.input_dir) if f.endswith('.csv'))
    
    print(f'File order: {[f[1] for f in files]}')

    with open(args.output_csv, 'w', encoding='utf-8') as out:
        for _, fn in files:
            path = os.path.join(args.input_dir, fn)

            with open(path, 'r', encoding='utf-8') as src:
                # strip newline, skip empty
                nums = [line.strip() for line in src if line.strip()]
                out.write(','.join(nums) + '\n')

if __name__ == '__main__':
    main()
