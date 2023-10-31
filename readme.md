# How to run
### Installing requirements
```pip install -r requirements.txt```
### Training the model
```train.ipynb``` <br>
Note: This will train the model on data from ```chatgpt_gen_date.json```
### Predict
```predict.py --conversation <str> --date <call's str> --label_type <str> --ensemble <bool>```
### Evaluation
```eval.py``` <br>
Note: This will evaluate the model on data from ```test_data.json```
### Make data to proper format: 
```format_data.py```

# Other details
1. Trained model weights are stored on hugging face.
2. Final model used - Finetuned FlanT5-large
3. Extra model weights (for model 1, 2, 3 (described in Approach_explanation.pdf)) uploaded to huggingface just in case. Although a few other changes would need to be made to the code to run using these weights.

# Approach Explanation: <br>
The details are in ```approach_explanation.pdf```