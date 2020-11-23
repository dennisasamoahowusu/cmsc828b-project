from __future__ import print_function
import nltk
from nltk.util import ngrams
import argparse
import numpy as np
from scipy.stats.mstats import gmean
import math
from utils import read_transfile
from utils import lee_transfile
import sentencepiece as spm
import operator
import sys
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--goldfile", help="gold file", required=False)
parser.add_argument("--predfile", help="pred file", required=False)
parser.add_argument("--spm_model")
args = parser.parse_args()

def spm_seg(sent, model, transform='enc'):
  sp = spm.SentencePieceProcessor()
  sp.Load(model)
  if transform == 'enc':
      spm_line = sp.EncodeAsPieces(sent)
  else:
        spm_line = sp.DecodePieces(sent.split())
  return ' '.join(spm_line).replace('▁', '')

def spm_seg_list(sent, model, transform='enc'):
  sp = spm.SentencePieceProcessor()
  sp.Load(model)
  if transform == 'enc':
      spm_line = sp.EncodeAsPieces(sent)
  else:
        spm_line = sp.DecodePieces(sent.split())
  return spm_line

#counts the number of ngrams in a sentence
def ngram_counts(sent,ngram_size):
  gram_dict = {}
  toks = sent.split()
  grams = list(ngrams(toks,ngram_size))
  for gram in grams:
    if gram not in gram_dict:
      gram_dict[gram] = 1
    else:
      gram_dict[gram] += 1
  return gram_dict

#turns dictionary of ngram counts to dictionary of ngram approximated probabilities
def ngram_probs(ngram_dict):
  gram_prob_dict = {}
  total_ngrams = float(sum(ngram_dict.values()))
  for ngram in ngram_dict:
    gram_prob_dict[ngram] = ngram_dict[ngram] / total_ngrams
  return gram_prob_dict

def tag_stems(raw_toks, mor_toks):
  index = 0
  tag_dict = {}
  temp_len = 0
  for tok in mor_toks:
    curr_word = raw_toks[index]
    c_len = len(curr_word)
    if tok in curr_word:
      tag_dict[tok] = index
      temp_len = temp_len + len(tok)
      if temp_len == c_len:
        index += 1
        temp_len = 0
  count_dict = {}
  for val in tag_dict.values():
    if val not in count_dict:
      count_dict[val] = 1
    else:
      count_dict[val] += 1
  split_dict = {}
  sub_cnt = 0
  for word in tag_dict:
    cnt = tag_dict[word]
    if count_dict[cnt] == 1:
      split_dict[word] = word
    else:
      sub_cnt +=1
      if sub_cnt == count_dict[cnt]:
        split_dict[word] = word
        sub_cnt = 0
      else:
        split_dict[word] = word + "##"
  return split_dict


#fetches count of ngram from dictionary; if not in dictionary returns 0
def get_cnt(dicta,key):
   value = dicta.get(key)
   if value == None:
      ans = 0
   else:
      ans = float(value)
   return ans

#computes modified precision according to ngram size
def modified_precision(sent,comp_sent,ngram_size):
  ngrams_sent = ngram_counts(sent,ngram_size)
  ngrams_comp_sent = ngram_counts(comp_sent,ngram_size)
  ngram_matches = {}
  for ngram in ngrams_sent:
      ngram_matches[ngram] = min(get_cnt(ngrams_sent,ngram),get_cnt(ngrams_comp_sent,ngram))
  numerator = sum(ngram_matches.values())
  denominator = sum(ngrams_sent.values())
  if numerator == 0 and denominator == 0:
    mod_prec = 0
  else:
    mod_prec = float(sum(ngram_matches.values()) / float(sum(ngrams_sent.values())))
  return mod_prec
  
def pairwise_BLEU(curr_sent,comp_sent,model):
  mod_prec_list = []
  curr_sent_seg = spm_seg(curr_sent,model,'enc')
  comp_sent_seg = spm_seg(comp_sent,model,'enc')
  for n in [1,2,3,4]:
    mod_prec = modified_precision(curr_sent_seg,comp_sent_seg,n)
    mod_prec_list.append(mod_prec)
    #print(mod_prec_list)
  with np.errstate(divide='ignore'):
    pw_BLEU = gmean(mod_prec_list)
  return pw_BLEU

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def discrepancy_score(sent_set,seg_model):
    #size_set = float(len(sent_set))
    size_set = 190
    score_factor = 1 / (size_set * (size_set -1) )
    score_per_sent = []
    size_set_range = range(len(sent_set))
    count = 0
    for sent in sent_set:
      curr_sent = sent
      multiple = sent_set[sent]
      pw_BLEU_per_sent = []
      for comp_sent in sent_set:
        #comp_sent_list = spm_seg_list(comp_sent, seg_model, 'enc')
        #curr_sent_list = spm_seg_list(curr_sent, seg_model, 'enc')
        #cc = SmoothingFunction()
        #pw_BLEU = sentence_bleu(curr_sent_list,comp_sent_list,smoothing_function=cc.method4)
        pw_BLEU = pairwise_BLEU(curr_sent,comp_sent,seg_model)
        pw_BLEU_per_sent.append( multiple * (1 - pw_BLEU))
      score_per_sent.append(sum(pw_BLEU_per_sent))
      count +=1
    DP_score = score_factor * sum(score_per_sent)
    return DP_score

def range_BLEU(sent_set,ref,model):
  running_BLEU = []
  #factor = float(1 / len(sent_set) )
  #refs_list = []
  #for ref in refs:
  #  refs_list.append(spm_seg_list(ref,model,'enc'))
  for sent in sent_set:
    sent_list = spm_seg_list(sent,model,'enc')
    cc = SmoothingFunction()
    sent_BLEU = sentence_bleu(ref,sent_list,smoothing_function=cc.method4)
    running_BLEU.append(sent_BLEU)
  avg_BLEU = float(sum(running_BLEU) / len(running_BLEU))
  return avg_BLEU

def main():
  #with open(args.goldfile) as f:
    #print("reading gold")
   # gold = read_transfile(f.readlines())

  #with open(args.predfile) as f:
  #  print("reading pred")
  #  pred = read_transfile(f.readlines())
  now = datetime.now().time()
  print(now)
  F = open('sample.txt')
  PRED = read_transfile(F.readlines())
  G = open('reference.txt')
  GOLD = read_transfile(G.readlines(),weighted=True)
  avg_BLEUs = {}
  count = 0
  for K in GOLD.keys():
    #refs = GOLD[K]
    ref = max(GOLD[K].items(), key = operator.itemgetter(1))[0]
    trans_set = PRED[K]
    avg_BLEU = round(range_BLEU(trans_set, ref,'subword.trg.model'),ndigits=4)
    avg_BLEUs[K] = avg_BLEU

  f = open('sample.txt')
  pred = lee_transfile(f.readlines())

  dp_score_list = {}
  for k, d in pred.items():
    trans_set = list(d)
    trans_dict = {}
    for item in trans_set:
      if item not in trans_dict:
        trans_dict[item] = 1
      else:
        trans_dict[item] += 1
    eprint(len(trans_dict))
    dp_score = round(discrepancy_score(trans_dict,'subword.trg.model'),ndigits=4)
    dp_score_list[k] = dp_score

  now = datetime.now().time()
  print(now)
  for K in GOLD.keys():
    DP = dp_score_list[K]
    AB = avg_BLEUs[K]
    print(str(DP) + '\t' + str(AB))

if __name__ == "__main__":
  main()
