import os
import google.generativeai as genai

# Load your Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Use Gemini Flash 2.0
model = genai.GenerativeModel('gemini-2.0-flash')

# South Indian real-world example question with more steps
question = """A tea shop in Bengaluru sells 120 cups of tea in the morning at 10 rupees each, 
and 80 cups in the evening at 12 rupees each. If the shop owner spends 500 rupees on milk 
and 200 rupees on tea leaves for the whole day, how much profit does the shop owner make in a day?"""

# Chain of Thought prompt
cot_prompt = f"""
Let's solve the following problem step by step:

Question: {question}

Let's think step by step.
"""

response = model.generate_content(cot_prompt)
print("Chain of Thought Answer:\n", response.text)