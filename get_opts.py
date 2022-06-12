import yaml
import re

s = ''
with open('train.py', 'r') as file:
    for line in file:
        # 要求 help 不要换行
        if 'parser.add_argument' in line:
            s += line

opts = []
defaults = []
comments = []
s = s.replace('\n', '')
lst = s.split('parser.add_argument')
for con in lst:
    tmp = con.strip()
    if tmp == '':
        continue
    # 表示两个及以上用{2,}
    tmp = re.sub(r'[ ]{2,}', '', tmp, 0)  # 去掉两个以上空格
    # 前向断言
    tmp = re.sub(r'(?<=,) ', '', tmp, 0)  # 去掉逗号后的空格
    tmpl = tmp.split(',')
    for ts in tmpl:
        if '--' in ts:
            opts.append(re.findall(r'--(.*)', ts)[0])
        if 'store_true' in ts:
            defaults.append('false')
        if 'default=' in ts:
            defaults.append(re.findall(r'=(.*)', ts)[0])
        if 'help' in ts:
            comments.append(re.findall(r'=(.*)', ts)[0])

y = dict()
for k, v, c in zip(opts, defaults, comments):
    y[k] = f'{v} # {c[1:-1]}'


with open('opts.yml', 'w') as f:
    yaml.dump(y, f)
