import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')

original_question = "What makes Dhoni a successful cricket captain?"
print("Original Question",original_question)

# Step-back prompt with NEW few-shot examples
step_back_prompt = f"""
You are an expert in general knowledge and reasoning. Your task is to take a step back 
and reframe specific questions into broader, more general ones. 
This helps in understanding the bigger picture. Here are some examples:

Q: Why do electric vehicles catch fire during summer in India?
Step-back Q: What factors cause electric vehicles to overheat or catch fire?

Q: How did Kerala handle the 2018 floods?
Step-back Q: How do governments typically respond to large-scale natural disasters?

Now, take a step back and reframe this:
Q: {original_question}
Step-back Q:"""

# Get step-back version
step_back_response = model.generate_content(step_back_prompt)
step_back_question = step_back_response.text.strip()

print("Step-back Question:", step_back_question)

# Now ask Gemini to answer both
final_prompt = f"""
Please provide a detailed answer combining both:
1. General (step-back) question: {step_back_question}
2. Specific question: {original_question}
"""

final_response = model.generate_content(final_prompt)
print("\nAnswer:\n", final_response.text)
