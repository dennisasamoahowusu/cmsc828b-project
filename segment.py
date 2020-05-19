import sentencepiece as spm
import argparse


def segment(model_path: str, input: str, output_file: str, no_sample: bool) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with open(input, "r") as f:
        contents = [x.strip("\n") for x in f.readlines()]

    piece_lengths = []
    for content in contents:
        if no_sample:
            pieces = sp.encode_as_pieces(content)
        else:
            pieces = sp.sample_encode_as_pieces(content, -1, 0.5)
        pieces_str = ""
        # pieces_length_str = ""
        for piece in pieces:
            pieces_str += piece + " "
            # pieces_length_str += str(len(piece)) + " "
        print(pieces_str)
        piece_lengths.append(len(pieces))

    with open(output_file, "w") as f:
        for x in piece_lengths:
            f.write('%s\n' % x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segments text using sentencepiece")
    parser.add_argument("--model", help="path to sentencepiece model", required=True)
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--ofile", help="input file", required=False)
    parser.add_argument("--nos", "--verbose", help="increase output verbosity",
                        action="store_true", required=False)
    args = parser.parse_args()

    segment(args.model, args.input, args.ofile, args.nos)
