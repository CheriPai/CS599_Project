from __future__ import division
import json
import numpy as np
import pandas as pd
from rankings import Rankings


val = pd.read_pickle("../data/val.pk")

correct = []
r = Rankings(1, "../data/train.pk")
i = 0
for item in val.iteritems():
    ranking = r.calculate_ranking(item[1])   
    if ranking[0].name == item[0]:
        correct.append(1)
    else:
        correct.append(0)

print(np.sum(correct) / len(correct))
