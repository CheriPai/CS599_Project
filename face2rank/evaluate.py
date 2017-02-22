from __future__ import division
import json
import numpy as np
import pandas as pd
from rankings import Rankings


val = pd.read_pickle("../data/val.pk")

correct = np.zeros((1680), dtype=np.int)
r = Rankings(5, "../data/train.pk")
for i, item in enumerate(val.iteritems()):
    ranking = r.calculate_ranking(item[1])   
    for j, celeb in enumerate(ranking):
        if ranking[j].name == item[0]:
            correct[i] = j+1
            break

print(np.bincount(correct))
