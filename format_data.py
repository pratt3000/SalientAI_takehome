import os
import json
import random
from datetime import timedelta, datetime
import en_core_web_lg

from constants import start_date, end_date, original_data_path, formatted_test_data_path

nlp = en_core_web_lg.load()

def get_random_date():

    # Calculate a random number of days to add to the start date
    random_days = random.randint(0, (end_date - start_date).days)

    # Create the random date by adding the random number of days to the start date
    random_date = start_date + timedelta(days=random_days)

    # Format the date as YYYY-MM-DD
    formatted_date = random_date.strftime("%Y-%m-%d")
    day = datetime.strptime(formatted_date, "%Y-%m-%d").strftime("%A")

    return formatted_date + ', ' + day


def get_time_entities(conversation):
    doc = nlp(conversation)

    standard_first_messages = ["Agent: Hi, I'm Taylor, calling from Westlake Financial on a recorded line."]
    if any(conversation.startswith(x) for x in standard_first_messages):
        conversation = '\n'.join(conversation.split('\n')[1:])

    entities = [(X.text, X.label_) for X in doc.ents if X.label_ == 'DATE' or X.label_ == 'ORDINAL']

    return entities


def format_data(data):

    new_data = []
    for id, ele in enumerate(data):
        conversation = ele['sample_input'].split('\n###\nTRANSCRIPT\n')[-1][:-4]
        auto_label = "NA" if get_time_entities(conversation) == [] else ""

        new_data.append({
            'id': id,
            'conversation_date': get_random_date(),
            'conversation': conversation,
            'label': auto_label,  
            'days_diff': 0
        })
    
    return new_data


if __name__ == "__main__":

    # To avoid overwriting the test data if it already exists
    if not os.path.exists(formatted_test_data_path):
        with open(original_data_path, 'r') as f:
            data = json.load(f)

        new_data = format_data(data)

        with open(formatted_test_data_path, 'w') as f:
            json.dump(new_data, f, indent=4)
    else:
        print("Test data already exists. Skipping...")
