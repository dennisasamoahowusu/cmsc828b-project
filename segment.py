import sentencepiece as spm
import argparse


def segment(model_path: str, input: str) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with open(input, "r") as f:
        contents = [x.strip("\n") for x in f.readlines()]

    for content in contents:
        pieces = sp.encode_as_pieces(content)
        pieces_str = ""
        for piece in pieces:
            pieces_str += piece + " "
        print(pieces_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segments text using sentencepiece")
    parser.add_argument("--model", help="path to sentencepiece model", required=True)
    parser.add_argument("--input", help="input file", required=True)
    args = parser.parse_args()

    segment(args.model, args.input)
