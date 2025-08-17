'''
Naive Bayes Pet Classifier (Dog vs Cat)
CS 441/541 Final Group Project
Jarren Calizo & Vivi Chen
'''

import math
import random
import re

# load data from pets.txt
def load_data(filename="pets.txt"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            label, text = line.split(" ", 1)
            data.append((label, text))
    return data

# clean text to lowercase, remove punctuation, split
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")

def tokenize(text):
    t = text.lower() # all lowercase
    t = PUNCT_RE.sub(" ", t) # replace punctuation with space
    t = SPACE_RE.sub(" ", t).strip() # clean-up extra space
    return [] if not t else t.split() # split into words


def preprocess(text, use_stop=True):
    toks = tokenize(text)
    if use_stop:
        toks = [w for w in toks if w not in STOPWORDS]
    return toks





def main():
    data = load_data("pets.txt")
    print(f"Loaded {len(data)} pet descriptions.")
    print("First 5 samples:")
    for sample in data[:5]:
        print(sample)
        
if __name__ == "__main__":
    main()