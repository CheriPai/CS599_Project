import json
import pandas as pd
import sys


with open(sys.argv[1]) as f:
    data = json.load(f)
    
df = pd.Series(data)

remove_items = [item[0] for item in df.iteritems() if len(item[1]) < 2]
df = df.drop(remove_items)

train = {}
val = {}
for item in df.iteritems():
    val[item[0]] = item[1][0]
    train[item[0]] = item[1][1:]

with open("train.json", "w+") as f:
    json.dump(train, f)

with open("val.json", "w+") as f:
    json.dump(val, f)
