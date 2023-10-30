import time
import warnings
warnings.filterwarnings("ignore")
import en_core_web_lg

from transformers import pipeline
from constants import model_path, label_type, base_prompts_list, agent_starter_dialogue


def check_entities_in_conversation(model, transcript):

    if transcript.startswith(agent_starter_dialogue):
        transcript = transcript[len(agent_starter_dialogue):]

    doc = model(transcript)
    entity_types = [X.label_ for X in doc.ents]
    if list(set(entity_types) & set(['DATE', 'CARDINAL'])) == []:
        return False
    return True

def get_predictions(model, model_entity_rec, conversation, date, label_type):

    base_prompt = base_prompts_list[label_type]
    prompt = conversation + "\nToday's Date: " + date + '\n' + base_prompt
    if check_entities_in_conversation(model_entity_rec, conversation):
        res = model(prompt)
    else:
        res = [{'generated_text': 'NA'}]
    
    return res


print(f"Loading model from {model_path}")
model_entity_rec = en_core_web_lg.load()
model_nlp = pipeline(model = model_path)

print("Running Prediction Engine\n")
start = time.time()
conversation = "Agent: Can you advise when you'll manage to make the payment? \nCustomer: I should be able to do it on Thursday next week."
date = "2022-01-01, Saturday"
res = get_predictions(model_nlp, model_entity_rec, conversation, date, label_type = label_type )

print("DATE: ", date)
print("CONVERSATION: ")
print(conversation)
print(f"RESULT: {res[0]['generated_text']}")
print(f"\nTime taken (CPU): {round(time.time() - start, 3)} seconds")