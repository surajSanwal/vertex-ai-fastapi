from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession

# Initialize the FastAPI app
app = FastAPI()

# Constants for configuration
PROJECT_ID = "mdz-cons-dev-genai-chat-svc"
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
        raise RuntimeError(f"Error in generating response: {str(e)}") from e

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
        raise HTTPException(status_code=500, detail=str(e)) from e

# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
