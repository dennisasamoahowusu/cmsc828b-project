import nltk
from nltk.util import ngrams
import argparse
from scipy.stats.mstats import gmean

parser = argparse.ArgumentParser()
parser.add_argument('--trans_file', help="list of outputs of model translation")
parser.add_argument('--ref_file', help="reference translation")
args = parser.parse_args()

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
  mod_prec = float(sum(ngram_matches.values()) / float(sum(ngrams_sent.values())))
  return mod_prec
  
def pairwise_BLEU(curr_sent,comp_sent):
  mod_prec_list = []
  for n in [1,2,3,4]:
    mod_prec = modified_precision(curr_sent,comp_sent,n)
    mod_prec_list.append(mod_prec)
  print(mod_prec_list)
  pw_BLEU = gmean(mod_prec_list)
  return pw_BLEU
    
def discrepancy_score(sent_set):
    size_set = float(len(sent_set))
    score_factor = 1 / (size_set * (size_set -1) )
    score_per_sent = []
    size_set_range = range(len(sent_set))
    for i in size_set_range:
      comp_set = [j for j in size_set_range if j != i]
      pw_BLEU_per_sent = []
      curr_sent = sent_set[i]
      for j in comp_set:
        comp_sent = sent_set[j]
        pw_BLEU = pairwise_BLEU(curr_sent,comp_sent)
        pw_BLEU_per_sent.append(1 - pw_BLEU)
      score_per_sent.append(sum(pw_BLEU_per_sent))
    DP_score = score_factor * sum(score_per_sent)
    return DP_score


def main():
      with open(args.trans_file, 'r') as f_trans, \
         open(args.ref_file, 'r') as f_ref:
             trans_data = f_trans.read().splitlines()
             ref_data = f_ref.read().splitlines()
             
             total_trans_set = trans_data + ref_data
             
             print(discrepancy_score(total_trans_set))

if __name__ == "__main__":
  main()
