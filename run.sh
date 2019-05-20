# for cars comment dataset
python main.py

python main.py -static=true

python main.py -static=true -non-static=true

python main.py -static=true -non-static=true -multichannel=true


# for du query dataset
python main.py -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20

python main.py -static=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20

python main.py -static=true -non-static=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20

python main.py -static=true -non-static=true -multichannel=true -dataset=../data/du_query/ -filter-num=50 -sen_len=20 -hidden_size=20


