from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
import os
import glob
from pathlib import Path
import shutil

load_dotenv()

client = OpenAI()

def read_file(file_path):
    """Read the contents of a file"""
    # print(f"🔨 Tool Called: read_file {file_path}")
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return f"Error: File {file_path} does not exist."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content):
    """Write content to a file, creating directories if needed"""
    # print(f"🔨 Tool Called: write_file {file_path}")
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def list_files(directory="."):
    """List files in a directory with a recursive option"""
    # print(f"🔨 Tool Called: list_files {directory}")
    try:
        files = glob.glob(os.path.join(directory, "**"), recursive=True)
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def create_directory(directory_path):
    """Create a directory structure"""
    # print(f"🔨 Tool Called: create_directory {directory_path}")
    try:
        os.makedirs(directory_path, exist_ok=True)
        return f"Successfully created directory: {directory_path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

def delete_file(file_path):
    """Delete a file"""
    # print(f"🔨 Tool Called: delete_file {file_path}")
    try:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
                return f"Successfully deleted file: {file_path}"
            else:
                return f"Error: {file_path} is not a file."
        else:
            return f"Error: File {file_path} does not exist."
    except Exception as e:
        return f"Error deleting file: {str(e)}"

available_tools = {
    "read_file": {
        "fn": read_file,
        "description": "Reads the contents of a file at the given file path"
    },
    "write_file": {
        "fn": write_file,
        "description": "Writes content to a file at the given file path, creating directories if needed"
    },
    "list_files": {
        "fn": list_files,
        "description": "Lists files in a directory, recursively by default"
    },
    "create_directory": {
        "fn": create_directory,
        "description": "Creates a directory structure (including nested directories)"
    },
    "delete_file": {
        "fn": delete_file,
        "description": "Deletes the specified file"
    }
}

system_prompt = """
    You are CodeAssist, an AI Coding Agent specialized in full-stack project development.
    You work on start, plan, action, observe mode to help users build and modify projects directly from the terminal.
    
    For the given user query and available tools, plan the step-by-step execution, select the relevant tool
    from the available tools, and based on the tool selection perform an action.
    Wait for the observation and based on the observation from the tool call, resolve the user query.
    
    CAPABILITIES:
    - Generate project structures (create directories and files)
    - Write code for both frontend and backend components
    - Read and modify existing code to add new features
    - Understand project context by examining files
    
    Rules:
    - Always follow the Output JSON Format
    - Always perform one step at a time and wait for the next input
    - When generating code, include complete implementation details
    - Carefully analyze the user query and existing code context
    - Before modifying existing files, always read them first
    - Use relative paths when appropriate
    - Always suggest best practices for the programming language or framework being used
    
    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}
    
    Available Tools:
    - read_file: Reads the contents of a file at the given file path
    - write_file: Writes content to a file at the given file path, creating directories if needed
    - list_files: Lists files in a directory, recursively by default
    - create_directory: Creates a directory structure (including nested directories)
    - delete_file: Deletes the specified file
    
    Example:
    User Query: Create a simple React app with a homepage
    Output: {{ "step": "plan", "content": "I'll help you create a simple React app with a homepage. First, I'll examine the workspace structure." }}
    Output: {{ "step": "action", "function": "list_files", "input": "." }}
    Output: {{ "step": "observe", "output": "[\".\", \"./src\", \"./src/components\"]" }}
    Output: {{ "step": "plan", "content": "Now I'll create a new homepage component." }}
    Output: {{ "step": "action", "function": "write_file", "input": "src/components/Homepage.js", "content": "import React from 'react'..." }}
    Output: {{ "step": "observe", "output": "Successfully wrote to src/components/Homepage.js" }}
    Output: {{ "step": "plan", "content": "Let's update the App.js file to include our new homepage" }}
    Output: {{ "step": "action", "function": "read_file", "input": "src/App.js" }}
    Output: {{ "step": "observe", "output": "import React from 'react'..." }}
    Output: {{ "step": "action", "function": "write_file", "input": "src/App.js", "content": "import React from 'react'..." }}
    Output: {{ "step": "output", "content": "I've created a React app with a customized homepage. The component has been added and the App.js has been updated to include it." }}
"""

messages = [
    { "role": "system", "content": system_prompt }
]

while True:
    user_query = input('\n> ')
    messages.append({ "role": "user", "content": user_query })

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })

        if parsed_output.get("step") == "plan":
            print(f"🧠 Planning: {parsed_output.get('content')}")
            continue
        
        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")
            
            # Special handling for write_file which needs two parameters
            if tool_name == "write_file":
                tool_input_parts = json.loads(tool_input) if isinstance(tool_input, str) and tool_input.startswith('{') else {"path": tool_input, "content": parsed_output.get("content", "")}
                file_path = tool_input_parts.get("path", "")
                content = tool_input_parts.get("content", "")
                output = available_tools[tool_name].get("fn")(file_path, content)
            else:
                if available_tools.get(tool_name, False) != False:
                    output = available_tools[tool_name].get("fn")(tool_input)
                else:
                    output = f"Error: Tool '{tool_name}' not found."
            
            messages.append({ "role": "assistant", "content": json.dumps({ "step": "observe", "output": output }) })
            continue
        
        if parsed_output.get("step") == "output":
            print(f"🤖 Output: {parsed_output.get('content')}")
            break

