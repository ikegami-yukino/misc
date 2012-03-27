#coding:utf-8
"""
Mozc辞書ファイルをMeCab辞書ファイルに変換

(MeCab辞書は、NAIST JDicでの利用を想定しています。
他の辞書での利用時には適宜書き換えてください)
"""

import sys, codecs, re, MeCab, optparse

# 変数初期設定
num = 0

# 正規表現ルール
re_hiragana = re.compile(u'[ぁ-ん]')

# MeCabオブジェクト生成(未知語表示指定)
tagger = MeCab.Tagger(" --unk-feature 未知語 -Odump")

# 単語がMeCabに既に登録されているかチェックする
def isUnregistered(word):
	i = 0
	n = tagger.parseToNode(word.encode("utf-8"))
	while n:
		if "未知語" in n.feature:
			return True
		i += 1
		n = n.next
	if i > 3:
		return True
	return False

# コスト計算
def getCost(word):
	result = tagger.parse(word.encode("utf-8")).rstrip('\n').decode('utf8').split('\n')
	return str(round(int(result[-1].split(' ')[14])*options.coefficient)).replace(".0","")

# 原形が*になってるものは表層形
def parsePos(pos, surface):
	features = pos.split(',')
	if features[-1] == u'*':
		return u','.join(features[:-1])+u','+surface
	return pos

# ひらがなをカタカナにする
def hira2kata(text):
	return re_hiragana.sub(lambda x: unichr(ord(x.group(0)) + 0x60), text)

if __name__ == '__main__':
	parser = optparse.OptionParser()
	parser.add_option("--mecab", dest="mecab_dic_path", help="MeCab-Dic path")
	parser.add_option("--mozc", dest="mozc_dic_path", help="mozc dic path")
	parser.add_option("-e", dest="encode", help="MeCab-Dic char code", default="utf8")
	parser.add_option("-o", dest="out_file", help="output file")
	parser.add_option("-l", dest="label", help="label", default="")
	parser.add_option("-c", dest="coefficient", help="cost coefficient", type="float", default=0.85)
	(options, args) = parser.parse_args()
	label = options.label.decode('utf8')

	# MozcとMeCabに共通する文脈IDを探す
	mecab_id_file = codecs.open(options.mecab_dic_path+'/left-id.def', "r", options.encode)
	mozc_id_file = codecs.open(options.mozc_dic_path+'/id.def', "r", 'utf8')
	out_file = codecs.open(options.out_file, "w", 'utf8')
	mecab_ids = [l.rstrip().split(' ') for l in mecab_id_file]
	mozc_ids = [l.rstrip().split(' ') for l in mozc_id_file]
	mecab_id_file.close()
	mozc_id_file.close()

	common_ids = []
	for mecab_id in mecab_ids:
		for mozc_id in mozc_ids:
			if mecab_id[1] == mozc_id[1]:
				common_ids.append({'mecab':mecab_id[0], 'mozc':mozc_id[0], 'pos':mecab_id[1]})
				break
	
	# 共通する文脈IDの単語をCSVに書きだす
	mozc_dics = [ codecs.open(options.mozc_dic_path+'/dictionary0'+str(i)+'.txt', "r", 'utf8') for i in range(0,9)]
	for mozc_dic in mozc_dics:
		for line in [ l.rstrip() for l in mozc_dic]:
			mozc_data = line.split('\t')
			for common_id in common_ids:
				if mozc_data[1] == common_id['mozc']:
					word = mozc_data[4]
					if isUnregistered(word):
						cost = getCost(word)
						pos = parsePos(common_id['pos'],word)
						yomi = hira2kata(mozc_data[0]).replace(u'・','')
						out_file.write(u"%s,%s,%s,%s,%s,%s,%s,%s\n" % (word, common_id['mecab'], common_id['mecab'], cost, pos, yomi, yomi, label))
						num += 1
						if num % 10000 == 0:
							print num
	out_file.close()
