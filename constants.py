from datetime import datetime

# File paths
original_data_path = "data/interview_date_training_data.json"
formatted_test_data_path = "data/data.json"

# Random date generation params (Generate a random date between two specified dates)
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Model params
model_id = "google/flan-t5-base"
model_name_on_hub = "Salient_ai" + model_id.split("/")[1]
model_path = "pratt3000/" + model_name_on_hub