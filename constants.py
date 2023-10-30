from datetime import datetime

# File paths
original_data_path = "data/interview_date_training_data.json"
formatted_test_data_path = "data/data.json"

# Random date generation params (Generate a random date between two specified dates)
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Data labels
base_prompts_list = {
    "label": "Given the above transcript and today's day and date, give me the date when the customer is expected to make their payment in the format 'dd/mm/yyyy'. Return 'NA' if its not possible to infer this information from the conversation. just return the date or NA and nothing else.", 
    "days_diff": "Given the above transcript and today's day and date, give me the number of days after which the customer will be able to pay. Return 0 if its not possible to infer this information from the conversation.Just return the number of days or 0 if not inferrable and nothing else."
}
agent_starter_dialogue = "Agent: Hi, I'm Taylor, calling from Westlake Financial on a recorded line. Unfortunately, we did not receive your monthly payment! Would you be able to make a payment today?\n"
