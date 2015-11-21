# -*- coding: utf-8 -*-
"""
Wikipediaのリダイレクト一覧を読みやすくJSON化

{
    そばの羽織': 'そば清',
    '徳島県立阿南養護学校': '徳島県立阿南支援学校',
    'オデッサ空港': 'オデッサ国際空港',
    '日動火災海上保険': '東京海上日動火災保険',
    'ジェイムス・モリソン': 'ジェームス・モリソン',
    'PNP予想': 'P≠NP予想',
    ...
}
"""
import gzip
import json
import os
import re
try:
    import urllib.request as urllib
except:
    import urllib

re_parentheses = re.compile("\((\d+),\d+,'?([^,']+)'?,[^\)]+\)")
URL_PAGES = ('https://dumps.wikimedia.org/jawiki/latest/'
             'jawiki-latest-page.sql.gz')
URL_REDIRECTS = ('https://dumps.wikimedia.org/jawiki/latest/'
                 'jawiki-latest-redirect.sql.gz')


def download():
    for url in (URL_PAGES, URL_REDIRECTS):
        print('download: %s' % url)
        urllib.urlretrieve(url, os.path.basename(url))


def extract_id_title(path):
    with gzip.GzipFile(path) as fd:
        id2title = dict(re_parentheses.findall(fd.read().decode('utf8')))
    return id2title


def extract_redirects(id2title, path):
    synonyms = {}
    with gzip.GzipFile(path) as fd:
        for (_from, to) in re_parentheses.findall(fd.read().decode('utf8')):
            if _from in id2title:
                synonyms[id2title[_from]] = to
    return synonyms


def write(synonyms, path):
    json.dump(synonyms, open(path, 'w'))


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
        write(synonyms, path='wikipedia_synonym.json')
    finally:
        claen()
