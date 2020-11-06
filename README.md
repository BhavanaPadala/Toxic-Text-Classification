# Toxic-Text-Classification

## Challenge:
In this model, a multi-headed model is built which is capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. Using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.
#### Disclaimer:
The dataset for this project contains text that may be considered profane, vulgar, or offensive
## Expected Result :
For each id in the test set, predict a probability for each of the six possible types of comment toxicity (toxic, severetoxic, obscene, threat, insult, identityhate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

id,toxic,severe_toxic,obscene,threat,insult,identity_hate <br>
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5<br>
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5<br>
etc.

### Data Set :
Provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:
toxic<br>
severe_toxic<br>
obscene<br>
threat<br>
insult<br>
identity_hate<br>
Create a model which predicts a probability of each type of toxicity for each comment.
