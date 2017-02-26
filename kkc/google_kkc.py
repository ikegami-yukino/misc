# -*- coding: utf-8 -*-
import copy
import json
try:
    from urllib.request import urlopen
    from urllib.parse import urlencode
except ImportError:
    from urllib import urlopen, urlencode
    input = raw_input


URL = 'http://www.google.com/transliterate'
PARAMS = {
    'langpair': 'ja-Hira|ja',
    'text': ''
}
MAX_PARAMS_LENGTH = 407


def _request(text):
    params = copy.copy(PARAMS)
    params['text'] = text
    url_params = urlencode(params)
    if len(url_params) > MAX_PARAMS_LENGTH:
        raise ValueError('number of given characters exceeds threshould')
    url = '%s?%s' % (URL, url_params)
    return urlopen(url)


def _convert(text):
    r = _request(text).read()
    if hasattr(r, 'decode'):
        r = r.decode('utf8')
    return json.loads(r)


def get_best_answer(text):
    bests = [candidates[0] for (raw, candidates) in _convert(text)]
    return ''.join(bests)


if __name__ == '__main__':
    while True:
        print(get_best_answer(input('> ')))
