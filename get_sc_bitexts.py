import argparse
import json


def create_bitext(map_file, prompts_file, translations_file, source_output_file, target_output_file):
    source_sentences = []
    target_sentences = []

    with open(map_file, encoding='utf-8') as data_file:
        mappings = json.loads(data_file.read())

    with open(prompts_file, encoding='utf-8') as data_file:
        prompts = [x.strip("\n") for x in data_file.readlines()]

    with open(translations_file, encoding='utf-8') as data_file:
        translations = [x.strip("\n") for x in data_file.readlines()]

    for mapping in mappings:
        prompt_index = mapping["prompt"]
        translation_index = mapping["trans"]
        sc = mapping["code"]

        prompt = prompts[prompt_index]
        translation = translations[translation_index]

        source = prompt
        target = "<CODE-" + str(sc) + ">" + "" + translation

        source_sentences.append(source)
        target_sentences.append(target)

    with open(source_output_file, 'w') as f:
        for sentence in source_sentences:
            f.write('%s\n' % sentence)

    with open(target_output_file, 'w') as f:
        for sentence in target_sentences:
            f.write('%s\n' % sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("get_sc_bitexts")
    parser.add_argument("--mfile", help="map file", required=True)
    parser.add_argument("--pfile", help="prompts file", required=True)
    parser.add_argument("--tfile", help="translations file", required=True)
    parser.add_argument("--sofile", help="source output file", required=True)
    parser.add_argument("--tofile", help="source output file", required=True)
    args = parser.parse_args()

    create_bitext(args.mfile, args.pfile, args.tfile, args.sofile, args.tofile)
