import time
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline


model_id = "google/flan-t5-base"
model_name_on_hub = model_name_on_hub = "Salient_ai" + model_id.split("/")[1]
model_path = "pratt3000/" + model_name_on_hub

model = pipeline(model = model_path)

sample = "MY name is Prathamesh"

start = time.time()
res = model(sample)

print(f"RESULT: {res[0]['generated_text']}")
print(f"Time taken (CPU): {round(time.time() - start, 3)} seconds")