#-*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import glob
import os
import re
import jctconv

re_asterisks = re.compile('[,\*]+$')


def parse_ipadic_id(ipa_id_def):
    result = {}
    for line in ipa_id_def.splitlines():
        (pos_id, pos) = line.split()
        original_pos = pos
        if ',BOS/EOS' in pos:
            pos = pos[:pos.rfind(',BOS/EOS')]
        pos = re_asterisks.sub('', pos)
        result[pos] = (pos_id, ','.join(original_pos.split(',')[:-1]))
    return result


def link_ipadic_mozc_pos(ipa_map, mozc_id_def):
    mozc_map = {}
    for line in mozc_id_def.splitlines():
        (pos_id, pos) = line.split()
        pos = re_asterisks.sub('', pos)
        pos = pos.replace('丁寧連用形', '連用形')
        if pos in ipa_map:
            mozc_map[pos_id] = ipa_map[pos]
    return mozc_map


def convert(mozc_map, mozc_dir, output_dir):
    with open(os.path.join(output_dir, 'mozc.csv'), 'w') as out_fd:
        for f in glob.glob(os.path.join(mozc_dir, 'src/data/dictionary_oss/dictionary*.txt')):
            with open(f) as in_fd:
                for l in in_fd:
                    l = l.decode('utf8').strip().split('\t')
                    (yomi, lid, rid, cost, surface) = l[:5]
                    if lid not in mozc_map:
                        continue
                    (new_id, pos) = mozc_map[lid]
                    yomi = jctconv.hira2kata(yomi)
                    line = ','.join([surface, new_id, new_id, '0', pos, surface, yomi, yomi])
                    line += '\n'
                    out_fd.write(line.encode('utf8', 'replace'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipadic', type=str, help='ipadic directory path')
    parser.add_argument('--mozc', type=str, help='mozc directory path')
    parser.add_argument('--out', type=str, help='output directory path')
    args = parser.parse_args()

    ipa_id_def = open(os.path.join(args.ipadic, 'right-id.def')).read().decode('eucjp')
    ipa_map = parse_ipadic_id(ipa_id_def)
    mozc_id_def = open(os.path.join(args.mozc, 'src/data/dictionary_oss/id.def')).read().decode('utf8')
    mozc_map = link_ipadic_mozc_pos(ipa_map, mozc_id_def)
    convert(mozc_map, args.mozc, args.out)
