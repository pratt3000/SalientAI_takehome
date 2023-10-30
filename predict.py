import time
import warnings
warnings.filterwarnings("ignore")
import en_core_web_lg

from transformers import pipeline


def check_entities_in_conversation(transcript):

    starter = "Agent: Hi, I'm Taylor, calling from Westlake Financial on a recorded line. Unfortunately, we did not receive your monthly payment! Would you be able to make a payment today?\n"
    if transcript.startswith(starter):
        transcript = transcript[len(starter):]

    doc = model_entity_rec(transcript)
    entity_types = [X.label_ for X in doc.ents]
    if list(set(entity_types) & set(['DATE', 'CARDINAL'])) == []:
        return False
    return True


model_id = "google/flan-t5-base"
model_name_on_hub = "Salient_ai" + model_id.split("/")[1]
model_path = "pratt3000/" + model_name_on_hub

model_entity_rec = en_core_web_lg.load()
model_nlp = pipeline(model = model_path)

conversation = "Agent: Can you advise when you'll manage to make the payment? \nCustomer: I should be able to do it on Thursday next week."
base_prompt = "Given the above transcript and today's day and date, give me the date when the customer is expected to make their payment in the format 'dd/mm/yyyy'. Return 'NA' if the conversation doesnt make sense and the customer hasnt promised of any dates. Just output the date in your response and nothing else."
date = "2022-01-01"
prompt = conversation + "\nToday's Date: " + date + '\n' + base_prompt

start = time.time()
if check_entities_in_conversation(conversation):
    res = model_nlp(prompt)
else:
    res = [{'generated_text': 'NA'}]

print(f"RESULT: {res[0]['generated_text']}")
print(f"Time taken (CPU): {round(time.time() - start, 3)} seconds")