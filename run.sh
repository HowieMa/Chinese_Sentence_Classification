# for cars comment dataset
python main.py -test-interval=50

python main.py -static=true -test-interval=50

python main.py -static=true -non-static=true -test-interval=50

python main.py -static=true -non-static=true -multichannel=true -test-interval=50


# for du query dataset
python main.py -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20 -test-interval=50

python main.py -static=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20 -test-interval=50

python main.py -static=true -non-static=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20 -test-interval=50

python main.py -static=true -non-static=true -multichannel=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20 -test-interval=50


