import json
from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel


with open("data/financial_phrasebank/financial_phrasebank_16_100_train.jsonl", "r") as f:
    train_data = []
    for line in f:
        train_data.append(json.loads(line))
        

# Load the model
data = MetaICLData(method="channel", max_length=1024, max_length_per_example=256)
model = MetaICLModel()
model.load("channel-metaicl")
model.cuda()
model.eval()

# Make a prediction for `input1`
input1 = "Both operating profit and net sales for the six-month period increased as compared to the corresponding period in 2007."
data.tensorize(train_data, [input1], options=["positive", "neutral", "negative"])
prediction = model.do_predict(data)[0]
print (prediction) # positive

# Make another prediction for `input2`
input2 = "The deal will have no significant effect on the acquiring company's equity ratio."
data.tensorize(train_data, [input2], options=["positive", "neutral", "negative"])
prediction = model.do_predict(data)[0]
print (prediction) # neutral