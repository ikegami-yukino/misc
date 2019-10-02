download:
	curl https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt > data/ptb.train.txt
	curl https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt > data/ptb.valid.txt
	curl https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt > data/ptb.test.txt

train:
	python train.py --train_path data/simple-examples/data/ptb.train.txt --dev_path data/simple-examples/data/ptb.valid.txt --delimiter " "
