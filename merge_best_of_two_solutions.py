#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='Take the solutions with the best k_i\'s of the two input files to create a new solution file.')
parser.add_argument('input_1', help='First input file')
parser.add_argument('input_2', help='Second input file')
parser.add_argument('output', help='Output file')

args = parser.parse_args()

out = []
with open(args.input_1, 'r', encoding='utf-8') as f1, open(args.input_2, 'r', encoding='utf-8') as f2:
    i = 1
    for line1, line2 in zip(f1, f2):
        k1 = len(line1.strip().split(','))
        k2 = len(line2.strip().split(','))
        out.append(line1 if k1 > k2 else line2)
        print(f"For denom={i}: {'k1' if k1 > k2 else 'k2'} was best. k1={k1}, k2={k2}")
        i *= 2

# doing it separately in case one of the files is also the output file
with open(args.output, 'w', encoding='utf-8') as f_out:
    f_out.write(''.join(out))
    