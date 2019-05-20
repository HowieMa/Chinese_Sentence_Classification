# Chinese_Sentence_Classification
Implementation of TextCNN, TextRNN, RCNN, FastText with Pytorch  
Pytorch 中文文本分类模型    

## Prerequisites
Python >= 3.5  
Pytorch >= 1.1.0  
torchtext  >= 0.3.1   
jieba >= 0.39  

## Data organizer 
Train and test data are organized in the following style  

`
Query, label
`

## Models
### TextCNN 
<br>
<img src="https://github.com/HowieMa/Chinese_Sentence_Classification/blob/master/src/img/TextCNN.png" />
<br>

### FastText
<br>
<img src="https://github.com/HowieMa/Chinese_Sentence_Classification/blob/master/src/img/fastText.png" />
<br>

### TextRNN 
<br>
<img src="https://github.com/HowieMa/Chinese_Sentence_Classification/blob/master/src/img/BiLSTM.jepg" />
<br>

### RCNN 
<br>
<img src="https://github.com/HowieMa/Chinese_Sentence_Classification/blob/master/src/img/RCNN.png" />
<br>



## Train 
In each folder, you could train model in 4 ways:  

Random Initialize Word Embedding   
`python main.py`

Use pre-trained Word Embedding  (freeze)  
`python main.py -static=true`

Use pre-trained Word Embedding  (not freeze)  
`python main.py -static=true -non-static=true`

Use pre-trained Word Embedding  (multi-channel : free + not freeze)   
`python main.py -static=true -non-static=true -multichannel=true`

### Word Embeddings 
[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)


# Reference 
[Text-Classification-Models-Pytorch ](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)  
[chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)


