from pathlib import Path
from typing import Annotated

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.ckpt"

KEYBOARD_MAP = [
    list("1234567890-="),
    list("qwertyuiop[]"),
    list("asdfghjkl;'"),
    list("zxcvbnm,./"),
]
KEY_POSITIONS = {
    character: (row, column)
    for row, characters in enumerate(KEYBOARD_MAP)
    for column, character in enumerate(characters)
}


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_lengths = torch.tensor([inputs.shape[1]], dtype=torch.int64)
        packed_input = rnn_utils.pack_padded_sequence(
            inputs,
            sequence_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output[:, -1, :])


class PredictionRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=500)]


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    accepted_text: str
    ignored_characters: list[str]


def encode_text(text: str) -> tuple[list[list[int]], str, list[str]]:
    coordinates: list[list[int]] = []
    accepted: list[str] = []
    ignored: list[str] = []

    for original_character in text:
        character = original_character.lower()
        position = KEY_POSITIONS.get(character)
        if position is None:
            ignored.append(original_character)
            continue
        coordinates.append([position[0], position[1]])
        accepted.append(character)

    return coordinates, "".join(accepted), ignored


def load_model() -> GRU:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    loaded_model = GRU()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    return loaded_model


model = load_model()
app = FastAPI(title="Writing Digit Detector", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    coordinates, accepted_text, ignored_characters = encode_text(payload.text)
    if not coordinates:
        raise HTTPException(
            status_code=422,
            detail="請輸入有效的鍵盤軌跡。",
        )

    inputs = torch.tensor(coordinates, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits = model(inputs)[0]
        probabilities = torch.softmax(logits, dim=0)
        prediction = int(torch.argmax(probabilities).item())

    return PredictionResponse(
        prediction=prediction,
        probabilities=[round(float(value), 6) for value in probabilities],
        accepted_text=accepted_text,
        ignored_characters=list(dict.fromkeys(ignored_characters)),
    )
