import time
import warnings
import argparse
warnings.filterwarnings("ignore")
import en_core_web_lg
from datetime import datetime, timedelta

from transformers import pipeline
from constants import base_prompts_list, agent_starter_dialogue


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


def add_days_to_date(res, date):
    num_days = res[0]['generated_text']
    temp_res = ""
    if str(num_days) == '0':
        temp_res = 'NA'
    else:
        temp_res = datetime.strptime(date.split(',')[0], '%Y-%m-%d') + timedelta(days=int(num_days))
        temp_res = temp_res.strftime('%Y-%m-%d')
    
    res[0]['generated_text'] = temp_res

    return res

def get_model_params(label_type):
    # Model params
    model_id = "google/flan-t5-large"
    model_name_on_hub = "Salient_ai" + model_id.split("/")[1] + "_" + label_type
    model_path = "pratt3000/" + model_name_on_hub

    return model_path

def predict(conversation, date, label_type):

    model_path = get_model_params(label_type)

    print(f"Loading model from {model_path}")
    model_entity_rec = en_core_web_lg.load()
    model_nlp = pipeline(model = model_path)

    print("Running Prediction Engine")
    start = time.time()
    res = get_predictions(model_nlp, model_entity_rec, conversation, date, label_type = label_type )

    res = add_days_to_date(res, date) if label_type == "days_diff" else res

    print(f"Time taken (CPU): {round(time.time() - start, 3)} seconds\n")
    return res[0]['generated_text']

if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--conversation', type=str, default="Agent: Can you advise when you'll manage to make the payment? \nCustomer: I should be able to do it on Thursday next week.")
    parser.add_argument('--date', type=str, default="2022-01-01, Saturday")
    parser.add_argument('--label_type', type=str, default="days_diff")
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()


    if args.ensemble:
        pred_1 = predict(args.conversation, args.date, "label")
        pred_2 = predict(args.conversation, args.date, "days_diff")

        if pred_1 == pred_2:
            pred = pred_1 + "(Confidence: HIGH)"
        else:
            pred = pred_1 + " or " + pred_2 + "(Confidence: LOW)"
    else:
        pred = predict(args.conversation, args.date, args.label_type)
        
    print("DATE: ", args.date)
    print("CONVERSATION: ")
    print(args.conversation)
    print(f"RESULT: {pred}")