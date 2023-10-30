## Approaches I tried/explored

### Entity recognition + Matching

We could build a very basic model based on entity recognition (scipy - en_core_web_lg) but this fails when the conversation gets longer than 2-3 turns and is unable to understand semantic complexity of dialogue exchange. So algorithms are out of the way, we need nlp.

### Hidden attribute models: https://dl.acm.org/doi/pdf/10.1145/3366424.3382089

Ham uses attention to form <person,attribute,value> but we'd need a dataset to train on for this to work.


### NLP models
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


# Notes to self.
flan-t5-small (77M) - this can work just thoda training lagega. Its giving out dates, just not the right dates.
base - same as small but changing prompt sometimes give right answers.
Lets work with Flan!

# Approach
1. Autolabelling - I am checking for entities in the conversation. If the data doesnt contain 'ORDINAL' or 'DATE' entities then there is no mention of any time/date/etc in the conversation and we can safely return NA. This has a 0 false positives when evaluated on the test set provided, so its a good way to quickly process outputs. I will use this in the main algorithm as well (given its so accurate.) If the conversation does have 'ORDINAL' or 'DATE' entities we can then pass it through the model. This trick is useful as I was instantly able to process 20/113 responses as NA.

## can explore predicting days instead of dates.
## can run 2 models and if same output then confidence high else low. Also NA wala hack add kiya toh accuracy can increase.