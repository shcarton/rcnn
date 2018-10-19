
import sys
import gzip
import numpy as np
import random

from sklearn import linear_model

np.set_printoptions(precision=2)

# expected number of samples
N = 100000

# use this many as heldout for evaluation
M = 10000

'''
So this selects beer reviews where the chosen aspect is most easily predictable from the other aspects?? Is that not
super biased?

'''

def get_correlation(rows, aspect):
    mat = np.vstack(rows).T
    return np.corrcoef(mat)[aspect]

aspect = int(sys.argv[1])
input_path = sys.argv[2]
output_path = sys.argv[3]
lst = [ ]
with gzip.open(input_path) as fin:
    for line in fin:
        y, sep, x = line.partition("\t")
        y = [ float(v) for v in y.split() ]
        xi = [ v for i,v in enumerate(y) if i != aspect ]
        yi = y[aspect]
        lst.append((xi,yi,y,line))

print "{} in total".format(len(lst))

regr = linear_model.LinearRegression()
regr.fit([ u[0] for u in lst ], [ u[1] for u in lst ])
print regr.coef_, type(regr.coef_)

pred = regr.predict([ u[0] for u in lst ])
data = [ (p-u[1], u[3], u[0], u[1], u[2]) for p, u in zip(pred, lst) ]
data = sorted(data, key=lambda x: x[0])


best = 1.0
loc = -1

rows = [ ]
i = 0
j = len(data)-1
while i < j:
    rows.append(data[i][-1])
    rows.append(data[j][-1])
    if len(rows)%10000 == 0:
        corr = get_correlation(rows, aspect)
        corr[aspect] = 0
        maxc = max(abs(max(corr)), abs(min(corr)))
        avgc = sum(corr)/4.0
        print "loc={}  gap={:.3f} {:.3f}  max={:.3f}  avg={:.3f}  {}".format(
                i+1,
                data[i][0],
                data[j][0],
                maxc,
                avgc,
                corr
            )
        if len(rows) >= N*0.8 and best >= maxc:
            best = maxc
            loc = i+1
        if len(rows) > N*2.0: break
    i+=1
    j-=1

print ""
print "best loc={}  max correlation={}".format(
        loc, best
    )

random.seed(1234)
selected = data[:loc] + data[-loc:]
random.shuffle(selected)

print get_correlation([ x[-1] for x in selected[:-M] ], aspect)
print get_correlation([ x[-1] for x in selected[-M:] ], aspect)

with gzip.open(output_path+".train.txt.gz", "w") as fout:
    for gap, line, xi, yi, y in selected[:-M]:
        fout.write(line)

with gzip.open(output_path+".heldout.txt.gz", "w") as fout:
    for gap, line, xi, yi, y in selected[-M:]:
        fout.write(line)


