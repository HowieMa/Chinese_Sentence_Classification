# Chinese_Sentence_Classification
Implementation of TextCNN, TextRNN, RCNN, FastText with Pytorch  


# Prerequisites
Python >= 3.5  
Pytorch >= 1.1.0  
torchtext  >= 0.3.1 


# Data organizer 
Train and test data are organized in the following style  

`
Query, label
`

e.g.  
`
你快休息吧我爱你小度,1 
`

# Train 
In each folder, you could train model in 4 ways:  

Random Initialize Word Embedding   
`python main.py`

Use pre-trained Word Embedding  (freeze)  
`python main.py -static=true`

Use pre-trained Word Embedding  (not freeze)  
`python main.py -static=true -non-static=true`

Use pre-trained Word Embedding  (multi-channel : free + not freeze)   
`python main.py -static=true -non-static=true -multichannel=true`



# Reference 
[Text-Classification-Models-Pytorch ](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)  

[chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)


