# How to run
### Installing requirements
```pip install -r requirements.txt```
### Training the model
```train.ipynb```
Note: This will train the model on data from ```chatgpt_gen_date.json```
### Predict
```predict.py```
### Evaluation
```eval.py```
### Make data to proper format: 
```format_data.py```

# Other details
1. Trained model weights are stored on hugging face.
2. Final model used - Finetuned FlanT5-large
3. Extra model weights (for model 1, 2, 3 (described in Approach_explanation.pdf)) uploaded to huggingface just in case. Although a few other changes would need to be made to the code to run using these weights.

# Approaches I tried/explored

### Algorithm based (Purely sequence matching)
I didnt try this approach as it would defeat the purpose of this assignment (i.e. testing proficiency in ML/AI/NLP)

### Entity recognition

We could build a very basic model based on entity recognition (scipy - en_core_web_lg) but this fails when the conversation gets longer than 2-3 turns and is unable to understand semantic complexity of dialogue exchange. So although we can use it, we will also need nlp.

### Hidden attribute models: https://dl.acm.org/doi/pdf/10.1145/3366424.3382089
Ham uses attention to form <person,attribute,value> but we'd need a bigger dataset to train on for this to work. 

### Using an API like ChatGPT
Just putting it here because in my experiments I got pretty much 95%+ accuracy using this.

### Entity recognition + NLP models
The models that we will work on will be instruction based open-source models. Ex - Flan T5, Vicuna, 
There are 3 approaches that we can take. 
1. Train a small model 
    - Basically take a pretrained model trained on general text (language understanding is vital) and finetune it for our task using RLHF or basic supervision.
    - Con - We'd need to make a dataset.
    - Pro - Smaller/Faster model.
2. Finetune a medium sized model
    - Take a medium sized model with supervised instruction fine-tuning done to it and finetune it.
    - Con - Model size is larger and inference time may be affected.
    - Pro - We'd need a smaller dataset
3. Use a pretrained model
    - Just use a big model.
    - Con - Model size will be huge
    - Pro - No dataset required
    - I tested this on 15 queries manually and this approach works with 100% accuracy for the models flan-t5-xxl, vicuna (although only flan-t5-xxl has a commercial license). Ofcourse testing with a bigger dataset is required to get a more reliable metric but this was enough to give me a basic idea.

# Final Approach

### Dataset creation:
1. I used ChatGPT to generate a small dataset of 100 samples and then curated it manually to make sure it was correct. (data/chatgpt_gen_date.json)
2. I also used ChatGPT to generate labels for the given sample set (by Salient) of ~110 samples and also reformatted it. This will be my test set. (data/test_data.json)

### Finetuning models
I trained 2 types of models. 
1. Target is the date in the form yyyy/mm/dd
2. Target is the number of days after the call's date that the user will pay the amount. 
3. I finetuned various variants of FlanT5 (i.e. small, base, large). Bigger models performed better as expected. I was unable to try the xl and xxl variants because of compute & time restrictions.

### Generating outputs
1. Entity recognition:
If the data doesnt contain 'ORDINAL' or 'DATE' entities then there is no mention of any time/date/etc in the conversation and we can safely return NA. Using this trick is useful as I was instantly able to process 20/113 responses as NA. False positives = 0.
2. Pass the conversation along with the prompt to get the output.
3. OPTIONAL: Ensembling - I have also created an ensemble of both types of models above to get a more reliable output.

### Evaluation
1. Exact accuracy
2. Standard deviation
3. Rouge1, Rouge2, Rougel, Rougelsum