import csv
import functools
import hashlib
import itertools
import json
import os
import random
import re
import sys
import unicodedata

from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple

FIELDSEP = "|"

def remove_punctuation(text: str) -> str:
     """
     Remove punctuations of several languages, including Japanese.
     """
     return "".join(
         itertools.filterfalse(lambda x: unicodedata.category(x).startswith("P"), text)
     )


def segment(text: str) -> str:
    """
    Character-level segmentation for Korean and Japanese
    """
    import regex

    # Chinese
    text = regex.sub(r"(\p{Han})", r" \1 ", text)
    # Korean
    text = regex.sub(r"(\p{Hangul})", r" \1 ", text)
    # Japenese
    text = regex.sub(r"(\p{Hiragana})", r" \1 ", text)
    text = regex.sub(r"(\p{Katakana})", r" \1 ", text)

    text = text.replace("  ", " ").strip()
    return text;


def makeID(text: str) -> str:
    textID = hashlib.md5(text.lower().encode()).hexdigest()
    return f"prompt_{textID}"

    
def read_trans_prompts(lines: List[str], lowercase=True) -> List[Tuple[str,str]]:
    """
    This reads a file in the shared task format, returns a list of Tuples containing ID and text for each prompt.
    """

    ids_prompts = []
    first = True
    for line in lines:
        if lowercase:
            line = line.strip().lower()
        else:
            line = line.strip()
        # in a group, the first one is the KEY. 
        # all others are part of the set. 
        if len(line) == 0:
            first = True
        else:
            if first:
                key, prompt = line.split(FIELDSEP)
                ids_prompts.append((key, prompt))
                first = False

    return ids_prompts

def read_transfile(lines: List[str], strip_punc=True, weighted=False, lowercase=True, length=0) -> Dict[str, Dict[str, float]]:
    """
    This reads a file in the shared task format, and returns a dictionary with prompt IDs as 
    keys, and each key associated with a dictionary of responses. 
    """
    data = OrderedDict()
    first = True
    options = {}
    key = ""
    keylen = 0
    for line in lines:
        if lowercase:
            line = line.strip().lower()
        else:
            line = line.strip()
        # in a group, the first one is the KEY. 
        # all others are part of the set. 
        if len(line) == 0:
            first = True
            if len(key) > 0 and len(options) > 0:
                if key in data:
                    print(f"Warning: duplicate sentence! {key}")
                if length == 0 or length == keylen:
                     data[key] = options
                options= {}
        else:
            if first:
                key, prompt = line.strip().split(FIELDSEP)
                keylen = len(prompt.split())
                first = False
            else:
                # allow that a line may have a number at the end specifying the weight that this element should take. 
                # this is controlled by the weighted argument.
                # gold is REQUIRED to have this weight.
                if FIELDSEP in line:
                    text, weight = line.strip().split(FIELDSEP)
                else:
                    text = line.strip()
                    weight = 1

                if strip_punc:
                    text = remove_punctuation(text)

                options[text] = float(weight)

    # check if there is still an element at the end.
    if len(options) > 0 and (length == 0 or length == keylen):
        data[key] = options

    return data
