# ReadMe

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server**:
   ```bash
   fastapi dev main.py
   ```

## API Endpoints

### 1. POST /upload-text

Uploads dialog data and saves messages from a specific speaker.

**Request Body**:
```json
{
  "dialog": [
    { "speaker": "User1", "content": "Привіт! Як справи?" },
    { "speaker": "User2", "content": "Все добре, дякую." },
    { "speaker": "User1", "content": "Підемо кави вип'ємо?" }
  ],
  "speaker": "User1"
}
```

**Response**:
```json
{
  "message": "Successfully saved messages for User1",
  "filepath": "./data/User1_messages.json"
}
```

### 2. POST /train

Trains the model for a specific speaker using LoRA.

**Request Body**:
```json
{
  "speaker": "User1"
}
```

**Response**:
```json
{
  "message": "Training completed successfully for User1",
  "model_path": "./output_lora/User1"
}
```

### 3. GET /docs

Swagger documentation

## Testing

Run the test script to verify the API functionality:

```bash
python test.py
```

## Project Structure

```
.
├── main.py          # Endpoints
├── parser.py        # Chat processing logic
├── train.py         # LoRA training logic
├── models.py        # Models
├── test.py          # Tests
└── requirements.txt # Dependencies
```

## LoRA Configuration

The training uses the following LoRA configuration:
- **r**: 8
- **alpha**: 16
- **target_modules**: ["q_proj", "v_proj"]
- **dropout**: 0.1
- **Training**: 1 epoch, batch size = 1

## Output

- **Messages**: Saved in `./data/<speaker>_messages.json`
- **Trained Models**: Saved in `./output_lora/<speaker>/`

## Notes

- The training is minimal (1 epoch, batch size = 1) to ensure the pipeline works
- The model uses 8-bit quantization for memory efficiency
- Training runs asynchronously to avoid blocking the API
- Only "User1" and "User2" are valid speaker values

## TODOs

- Add tuning possible on multiple devices (e.g. few gpus)
- Add wandb integration
- Add evaluation
- Make dataset formating dependent on the model
