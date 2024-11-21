# FastAPI and Vertex AI Integration Guide

## Overview

This guide provides a step-by-step approach to setting up a FastAPI application that interacts with Google Cloud's Vertex AI. The application processes user prompts and returns generated responses using Vertex AI's capabilities. The setup is tailored for MacOS users.

## Prerequisites

Google Cloud Account: Ensure you have access to Google Cloud and Vertex AI.
Google Cloud SDK: Installed on your MacOS for authentication and project management.
Python 3.x: Installed on your system.
FastAPI and Uvicorn: For building and running the API.

### Setup Instructions

#### Step 1: Install Google Cloud SDK

Install the Google Cloud SDK using Homebrew:

```bash
brew install --cask google-cloud-sdk
```

#### Step 2: Initialize Google Cloud SDK

Initialize the SDK and authenticate your account:

```bash
gcloud init
```

#### Step 3: Set Up Application Default Credentials

Authenticate your application to use Google Cloud services:

```bash
gcloud auth application-default login
```

#### Step 4: Create a Python Virtual Environment

Set up a virtual environment to manage dependencies:

```bash
mkdir vertex-ai-fastapi
cd vertex-ai-fastapi
python3 -m venv venv
source venv/bin/activate
```

#### Step 5: Install Required Python Packages

Install FastAPI, Uvicorn, and the Vertex AI client library:

```bash
pip install fastapi uvicorn google-cloud-aiplatform
```

#### Step 6: FastAPI Application Code

Create a file named main.py with the following code:

```python

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

# Initialize the FastAPI app
app = FastAPI()

# Constants for configuration
PROJECT_ID = "PROJECT_ID"
LOCATION = "us-central1"
MODEL_NAME = "gemini-1.5-flash-002"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel(MODEL_NAME)
chat_session = model.start_chat()

def get_chat_response(chat: ChatSession, prompt: str) -> str:
    """
    Sends a prompt to the chat session and returns the response.

    :param chat: The chat session object.
    :param prompt: The user-provided prompt.
    :return: The generated response as a string.
    """
    try:
        responses = chat.send_message(prompt, stream=True)
        text_response = [chunk.text for chunk in responses]
        return "".join(text_response)
    except Exception as e:
        raise RuntimeError(f"Error in generating response: {str(e)}")

# Define a request model using Pydantic
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to handle prompt requests
@app.post("/generate-response")
async def generate_response(request: PromptRequest):
    """
    Endpoint to generate a response from the Vertex AI model based on the user prompt.

    :param request: The request body containing the prompt.
    :return: The generated response.
    """
    try:
        response = get_chat_response(chat_session, request.prompt)
        return {"response": response}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

```

#### Step 7: Run the FastAPI Application

Start the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

### Usage Example

To test the API, use a tool like curl or Postman to send a POST request:

```bash
curl -X POST "http://127.0.0.1:8000/generate-response" -H "Content-Type: application/json" -d '{"prompt": "Hello, Vertex AI!"}'
```

## Future Enhancements

*Authentication*: Implement authentication mechanisms to secure the API.

*Logging*: Add logging to monitor API usage and errors.

*Testing*: Develop unit and integration tests to ensure functionality.

*Deployment*: Consider deploying the application on a cloud platform for scalability.

## Conclusion

This guide provides a comprehensive overview of setting up a FastAPI application integrated with Vertex AI. By following these steps, you can create a robust and scalable solution for processing user prompts and generating responses using Google's AI capabilities.
