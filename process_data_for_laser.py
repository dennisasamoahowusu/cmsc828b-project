import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fpath", help="Train file (in original format)")
    parser.add_argument("prompt_output_fpath", help="Prompt file")
    parser.add_argument("translation_output_fpath", help="Translation file")
    args = parser.parse_args()

    # prepare path names
    outdir = Path(args.prompt_output_fpath).parent
    outdir.mkdir(exist_ok=True)
    map_outpath = Path(outdir, f"map.json")

    # split up prompts and translations into different text files
    idx_map = []
    prompt_idx, trans_idx = 0, 0
    with open(args.input_fpath, mode="r", encoding="utf-8") as infile,\
        open(args.prompt_output_fpath, mode="w", encoding="utf-8") as en_outfile,\
        open(args.translation_output_fpath, mode="w", encoding="utf-8") as fo_outfile,\
        open(map_outpath, mode="w") as map_outfile:
        for line in infile:
            if not line.strip():
                prompt_idx += 1
                continue
            
            line_begin, line_end = line.strip().split('|')
            
            if line_begin.startswith("prompt"):
                en_outfile.write(f"{line_end}\n")
            else:
                fo_outfile.write(f"{line_begin}\n")
                idx_map.append({'prompt': prompt_idx, 'trans': trans_idx})
                trans_idx += 1

        json.dump(idx_map, map_outfile)


            
            