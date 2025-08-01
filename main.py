from models import UploadTextRequest, TrainRequest
from asyncio import AbstractEventLoop
from huggingface_hub import login
from dotenv import load_dotenv
from parser import ChatParser
from train import LoRATrainer
from fastapi import FastAPI
import asyncio
import os


# Initializations
load_dotenv()
assert (HF_API_KEY := os.getenv("HF_API_KEY")), "No HF_API_KEY in .env"
login(HF_API_KEY)

# "meta-llama/Meta-Llama-3-8B-Instruct"
trainer: LoRATrainer = LoRATrainer("gpt2", "./output_lora")
parser: ChatParser = ChatParser("./data")
app: FastAPI = FastAPI()


@app.post("/upload-text")
async def upload_text(request: UploadTextRequest):
    """
    Upload dialog data and save messages from a specific speaker
    """
    path: str = parser.save(request.dialog, request.speaker)
    return {
        "message": f"Successfully saved messages for {request.speaker}",
        "messages_path": path,
    }


@app.post("/train")
async def train(request: TrainRequest):
    """
    Train the model for a specific speaker using LoRA
    """
    # Run the training in a thread pool
    loop: AbstractEventLoop = asyncio.get_event_loop()
    path: str = await loop.run_in_executor(
        None,
        lambda: trainer.train_model(
            request.speaker, parser.load(request.speaker)
        ),
    )
    return {
        "message": f"Training completed successfully for {request.speaker}",
        "model_path": path,
    }
