# -*- coding: utf-8 -*-
"""
Wikipediaのリダイレクト一覧からElasticsearchの類義語辞書を生成

e.g.
ベイクドチーズケーキ,CHEESE CAKE,チーズケーキ,焼きたてチーズケーキ,レアチーズケーキ
ターンバック,ターン・バック
黒崎バイパス,黒崎道路
"""
from __future__ import unicode_literals
import codecs
from collections import defaultdict
import gzip
import os
import re
try:
    import urllib.request as urllib
except:  # for Python 2.X
    import urllib
    chr = unichr

re_parentheses = re.compile("\((\d+),\d+,'?([^,']+)'?,[^\)]+\)")
re_title_brackets = re.compile('_\([^\)]+\)$')
IGNORE_PREFIX = ('削除依頼/', '検証/', '進行中の荒らし行為/', '井戸端/', 'WP:',
                 '利用者:', 'User:', 'ウィキプロジェクト', 'PJ:')
IGNORE_SUBSTR = ('Wikipedia:', 'Template:', 'Listes:', '過去ログ:', 'ファイル:', '画像:',
                 'Section:')
URL_PAGES = ('https://dumps.wikimedia.org/jawiki/latest/'
             'jawiki-latest-page.sql.gz')
URL_REDIRECTS = ('https://dumps.wikimedia.org/jawiki/latest/'
                 'jawiki-latest-redirect.sql.gz')
HIRAGANA = set(map(chr, range(12353, 12353+86)))
KATAKANA = set(map(chr, range(12449, 12449+90)))


def download():
    for url in (URL_PAGES, URL_REDIRECTS):
        print('download: %s' % url)
        urllib.urlretrieve(url, os.path.basename(url))


def extract_id_title(path):
    with gzip.GzipFile(path) as fd:
        id2title = dict(re_parentheses.findall(fd.read().decode('utf8')))
    return id2title


def normalize(title):
    title = re_title_brackets.sub('', title)
    return title.replace('_', ' ')


def is_valid_title(title):
    if (title.startswith(IGNORE_PREFIX) or title.endswith('の一覧') or
            title in HIRAGANA or title in KATAKANA):
        return False
    return not any(s in title for s in IGNORE_SUBSTR)


def extract_redirects(id2title, path):
    synonyms = defaultdict(set)
    with gzip.GzipFile(path) as fd:
        for (from_id, to) in re_parentheses.findall(fd.read().decode('utf8')):
            to = normalize(to)
            if from_id in id2title:
                _from = id2title[from_id]
                _from = normalize(_from)
                if (_from == to or
                        not is_valid_title(_from) or not is_valid_title(to)):
                    continue
                if _from in synonyms:
                    synonyms[to] |= synonyms[_from]
                    del synonyms[_from]
                synonyms[to].add(_from)
    return synonyms


def write(synonyms, path):
    print('write: %s' % path)
    with codecs.open(path, 'w', encoding='utf8') as fd:
        for (word, words) in synonyms.items():
            words.add(word)
            fd.write('%s\n' % (','.join(words)))


def claen():
    for filename in ('jawiki-latest-page.sql.gz',
                     'jawiki-latest-redirect.sql.gz'):
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    try:
        download()
        id2title = extract_id_title(path='jawiki-latest-page.sql.gz')
        synonyms = extract_redirects(id2title,
                                     path='jawiki-latest-redirect.sql.gz')
        write(synonyms, path='wikipedia_synonym.txt')
    finally:
        claen()
