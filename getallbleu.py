import os
import sys
import io

def getmax(fn):
    if not os.path.exists(fn):
        print(" not exist".format(fn))
        exit()
    best = 0.

    with io.open(fn, 'r', encoding='utf8', newline='\n') as src:
        for line in src:
            linewords = line.strip().split()
            length = len(linewords)
            if length == 14:
                bleu = linewords[8]
                bleu = float(bleu[:-1])
                if bleu > best:
                    best = bleu
    return best

tradeoffs = [0.0,0.05, 0.1, 0.15, 0.2 ]
tradeoff_bleu = []
for tradeoff in tradeoffs:
    fn = 'checkpoints/wmt14_en2de_{tradeoff}/bleu.txt'.format(tradeoff=tradeoff, )
    bleu = getmax(fn)
    tradeoff_bleu.append(bleu)
print('tradeoff {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}'.format(*tradeoffs))
print('    bleu {:^5.2f} {:^5.2f} {:^5.2f} {:^5.2f} {:^5.2f}'.format(*tradeoff_bleu))
