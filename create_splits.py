#!/usr/bin/env python3

import random
random.seed(8) #for consitency in our train/test split

import sys
sys.path.append("..") # Adds higher directory to python modules path.


from utils import read_transfile, read_trans_prompts
from collections import defaultdict



langs = "hu ja ko pt vi".split()

data = defaultdict(dict)
all_prompts = {}
shared_prompts = {}
prompts = defaultdict(dict)

#data_path="/Users/hudakhayrallah/Downloads/staple-2020-train/"
data_path=""


# Build list of prompts shared across languages
for lang in langs:
    trainset = read_transfile(open(f"{data_path}en_{lang}/train.gold").readlines(), weighted=True, strip_punc=False)
    lang_prompts = read_trans_prompts(open(f"{data_path}en_{lang}/train.gold").readlines())
    for key, prompt in lang_prompts:
        prompts[lang][key] = prompt

    for prompt, translations in trainset.items():
#        print(f"{lang} {prompt} {prompts[lang][prompt]}")
        data[lang][prompt] = translations
        if prompt in all_prompts:
            shared_prompts[prompt] = 1
        all_prompts[prompt] = 1

for lang in langs:
    not_shared = list(filter(lambda x: x not in shared_prompts, data[lang].keys()))
    print(f"{lang}: {len(data[lang].keys())} not shared: {len(not_shared)}")

# Build test sets, seeded with shared prompts
for lang in langs:
    test_split = random.sample(list(filter(lambda x: x not in shared_prompts, data[lang].keys())), 500)

   
    print(f"Writing files for {lang}", file=sys.stderr)
    with open(f"{data_path}en_{lang}/split.test", "w") as out:
        for prompt in test_split:
            print("|".join([prompt, prompts[lang][prompt]]), file=out)
            for translation, value in data[lang][prompt].items():
                print(translation, value, sep="|", file=out)
            print(file=out)


    #further splitting in to 3 sub test splits. of sizes 100, 100, 300
    test_split0, test_split1, test_split2 = test_split[:100], test_split[100:200],  test_split[200:]



    with open(f"{data_path}en_{lang}/split.test0", "w") as out:
        for prompt in test_split0:
            print("|".join([prompt, prompts[lang][prompt]]), file=out)
            for translation, value in data[lang][prompt].items():
                print(translation, value, sep="|", file=out)
            print(file=out)


    with open(f"{data_path}en_{lang}/split.test1", "w") as out:
        for prompt in test_split1:
            print("|".join([prompt, prompts[lang][prompt]]), file=out)
            for translation, value in data[lang][prompt].items():
                print(translation, value, sep="|", file=out)
            print(file=out)

    with open(f"{data_path}en_{lang}/split.test2", "w") as out:
        for prompt in test_split2:
            print("|".join([prompt, prompts[lang][prompt]]), file=out)
            for translation, value in data[lang][prompt].items():
                print(translation, value, sep="|", file=out)
            print(file=out)



    with open(f"{data_path}en_{lang}/split.train", "w") as out:
        for prompt in data[lang].keys():
            if prompt not in test_split:
                print("|".join([prompt, prompts[lang][prompt]]), file=out)
                for translation in data[lang][prompt]:
                    print(translation, value, sep="|", file=out)
                print(file=out)
