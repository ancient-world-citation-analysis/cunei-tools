"""
Command-line interface for cunei-tools.

Usage:
    # Segment unsegmented Unicode cuneiform
    cunei-seg --model model.json --input text.txt --output segmented.txt
    cunei-seg --model model.json --text "𒀀𒈾𒁀𒌅"

    # Convert transliteration to Unicode
    cunei-conv --mode to-unicode --text "a-na ba-ab"
    cunei-conv --mode to-latin --text "𒀀𒈾"
    cunei-conv --mode to-unicode --input texts.txt --output converted.txt
"""

import argparse
import sys


def seg_main():
    parser = argparse.ArgumentParser(
        description='cunei-seg: Word boundary segmentation for cuneiform'
    )
    parser.add_argument('--model', required=True, help='Path to trained model JSON')
    parser.add_argument('--text', help='Single text to segment')
    parser.add_argument('--input', help='Input file (one text per line)')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--threshold', type=float, help='Override threshold')
    args = parser.parse_args()

    from cunei_tools import CuneiSeg
    seg = CuneiSeg.load(args.model)

    if args.threshold:
        seg.threshold = args.threshold

    if args.text:
        print(seg.segment(args.text))
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f]
        results = seg.segment_batch(lines)
        out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
        for r in results:
            out.write(r + '\n')
        if args.output:
            out.close()
    else:
        parser.print_help()


def conv_main():
    parser = argparse.ArgumentParser(
        description='cunei-conv: Cuneiform transliteration converter'
    )
    parser.add_argument('--mode', required=True, choices=['to-unicode', 'to-latin'],
                        help='Conversion direction')
    parser.add_argument('--dict', help='Path to sign dictionary JSON')
    parser.add_argument('--text', help='Single text to convert')
    parser.add_argument('--input', help='Input file (one text per line)')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--lang', choices=['akk', 'sux', 'elx'],
                        help='Language for specialized normalization')
    args = parser.parse_args()

    from cunei_tools import CuneiConv

    if args.dict:
        conv = CuneiConv.load_from_file(args.dict)
    else:
        conv = CuneiConv()
        conv.load_sign_lists()

    def convert(text):
        if args.mode == 'to-unicode':
            return conv.to_unicode(text, lang=args.lang)
        else:
            return conv.to_latin(text)

    if args.text:
        print(convert(args.text))
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f]
        out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
        for line in lines:
            out.write(convert(line) + '\n')
        if args.output:
            out.close()
    else:
        parser.print_help()


if __name__ == '__main__':
    print("Use 'cunei-seg' or 'cunei-conv' commands.")
