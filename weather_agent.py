from dotenv import load_dotenv
from openai import OpenAI
import json
import os

load_dotenv()

client = OpenAI()

def get_weather(city: str):
    return f"The weather in {city} is 28 degree C."

system_prompt = """
    You are an helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}


    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interested in weather data of Bengaluru" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "Bengaluru" }}
    Output: {{ "step": "observe", "output": "28 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for Bengaluru seems to be 28 degrees." }}

"""

completion = client.chat.completions.create(
  model="gpt-4o",
  response_format={"type": "json_object"},
  messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": "what is the weather of Bengaluru"},
      {"role": "user", "content": json.dumps({"step": "plan", "content": "The user is interested in weather data of Bengaluru"})},
      {"role": "user", "content": json.dumps({"step": "plan", "content": "From the available tools I should call get_weather"})},
      {"role": "user", "content": json.dumps({"step": "action", "function": "get_weather", "input": "Bengaluru"})},
      {"role": "user", "content": json.dumps({"step": "observe", "output": get_weather("Bengaluru")})},
      {"role": "user", "content": json.dumps({"step": "output", "content": "The weather for Bengaluru seems to be 28 degrees."})}
  ]
)

print(completion.choices[0].message.content)
