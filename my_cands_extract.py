#!/usr/bin/env python3

import argparse
import debpe
import hashlib
import sys

from utils import FIELDSEP, makeID, remove_punctuation


def main(args):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 
    """
    cands = []
    seen_cands = set()
    current_source = None
    for line in args.infile:
        tokens = line.strip().split("\t")
        if line.startswith("S-"):
            # it's hard to have fairseq pass prompt ids through the training/evaluation process
            # so we resort to regenerating ids based on the prompt text.
            # we have to be careful that the text is *exactly* the same, or the id generation will be wrong.
            current_source = debpe.clean(tokens[1]) if not args.no_clean else tokens[1]
            textID = makeID(current_source)
            print(f"\n{textID}{FIELDSEP}{current_source}", file=args.outfile)
            cands = []
            seen_cands.clear()
        elif line.startswith("T-"):
            pass
        elif line.startswith("H-") and len(tokens) == 3 and not '-inf' in line:
            score = float(tokens[1])
            if len(cands) == 0:
              top_score = score
              if args.threshold != 0.0:
                prompt_threshold = (-1.0 * args.threshold) + top_score
            # this is the prediction, there may be many of these.
            if ((args.candlimit == -1 or len(cands) < args.candlimit) and \
                (args.threshold == 0.0 or score > prompt_threshold)):
                
                hyp = debpe.clean(tokens[2]) if not args.no_clean else tokens[2]
                hyp = hyp.lower()

                # remove language code if present
                if hyp.startswith("<") and len(hyp) >= 4 and hyp[3] == '>':
                    hyp = hyp[5:]

                if args.remove_punctuation:
                    hyp = remove_punctuation(hyp)

                if not hyp in seen_cands:
                    print(hyp, file=args.outfile)

                    cands.append(hyp)
                    seen_cands.add(hyp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This processes the output of fairseq-generate so that it can be scored with sacrebleu and so that it has the shared task format. ")
    parser.add_argument("--infile", "-i", type=argparse.FileType("r"), default=sys.stdin,
                        help="Name of output file from fairseq-generate")
    parser.add_argument("--outfile", "-o", type=argparse.FileType("w"), default=sys.stdout, 
                        help="Name of desired output file. This will be the shared task format file.")
    parser.add_argument("--candlimit", "-c", help="Max number of candidates to put in file (default is -1, meaning all)", type=int, default=-1)
    parser.add_argument("--no-clean", action="store_true", help="Don't remove subword artifacts")
    parser.add_argument("--remove-punctuation", "-r", action="store_true", help="Remove punctuation")
    parser.add_argument("--lang", "-l", default=None, help="Target language")
    parser.add_argument("--threshold", "-t", default=0.0, type = float)
    args = parser.parse_args()

    main(args)
