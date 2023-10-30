from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")
import en_core_web_lg

from transformers import pipeline
from constants import model_path, label_type, base_prompts_list, agent_starter_dialogue

from datetime import datetime

def date_difference_in_days(date_str1, date_str2):
    
    if label_type == "days_diff":
        return abs(int(date_str1) - int(date_str2))
    
    # Define the format of the date string
    date_format = "%Y-%m-%d"

    # Parse the date strings into datetime objects
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)

    # Calculate the difference in days
    delta = date2 - date1
    return abs(delta.days)


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


with open("data/test_data.json", 'r') as f:
    data = json.load(f)

print(f"Loading model from {model_path}")
print("label_type = ", label_type)

model_entity_rec = en_core_web_lg.load()
model_nlp = pipeline(model = model_path)

same = 0
cur_dist = 0
num_exceptions= 0
for id, d in tqdm(enumerate(data)):
    res = get_predictions(model_nlp, model_entity_rec, d["conversation"], d['conversation_date'], label_type = label_type )
    
    if res[0]['generated_text'] == str(d[label_type]):
        same += 1
        cur_dist += 0
    elif res[0]['generated_text'] == 'NA' or str(d[label_type]) == 'NA':
        cur_dist += 10 # arbitrary 10 day error added
    else:
        try:
            cur_dist += date_difference_in_days(res[0]['generated_text'], str(d[label_type]))
        except Exception as e:
            print(e)
            print(res[0]['generated_text'], str(d[label_type]))

print("ACCURACY (test) = ", same/(len(data) - num_exceptions))
print("avg_deviation (test) = ", cur_dist/(len(data) - num_exceptions))