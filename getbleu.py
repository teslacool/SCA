import argparse
import sys
import io
import os
import time
parser = argparse.ArgumentParser()
parser.add_argument('dirpath', type=str, help='checkpoints path')
parser.add_argument('start_step', type=int,  help='start step')
parser.add_argument('data_path', type=str,  help='as the name implies')
parser.add_argument('sleeptime', type=float, )
params = parser.parse_args()
dirpath = params.dirpath
minim_step = params.start_step
data_path = params.data_path

def flush_bleu_file(file):
    with io.open(file, 'w', encoding='utf8', newline='\n', errors='ignore') as src:
        pass

def write_bleu_file(file, step):
    with io.open(file, 'a', encoding='utf8', newline='\n', errors='ignore') as src:
        src.write('\n{:03d}'.format(step) + '\t' )

python_exe = sys.executable
bleu_file = os.path.join(dirpath, 'bleu.txt')
# flush_bleu_file(bleu_file)
python_file = 'generate.py'
grep_str = 'Generate test with beam'
cnt = 0
while True:
    checkpoints_file = os.listdir(dirpath)
    checkpoints_file.sort()
    # iterate each ckt file
    for ckt in checkpoints_file:
        cnt = 0
        if ckt.startswith('checkpoint') and ckt.endswith('.pt'):
            step = ckt.split('.')[0][10:]
            if not step.isdigit():
                # last of best
                continue
            step = int(step)
            ckt_path = os.path.join(dirpath, ckt)
            already_ckt = ckt_path + '.already'
            if os.path.exists(already_ckt):
                continue
            if step < minim_step:
                continue
            write_bleu_file(bleu_file, step)
            cmd = '{} {} {} --path {} '.format(python_exe, python_file, data_path, ckt_path) + \
                ' --batch-size 128 --beam 5 --remove-bpe --quiet | tee /dev/stderr | ' + \
                "grep -P '{}' >> {}".format(grep_str, bleu_file)
            os.system(cmd)
            cmd = 'touch {}'.format(already_ckt)
            os.system(cmd)
    print('sleep for {:03.1f} hours'.format(params.sleeptime))
    time.sleep(60 * 60 * params.sleeptime)
    cnt += 1
    if cnt > 3:
        break
