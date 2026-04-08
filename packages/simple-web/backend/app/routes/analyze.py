from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..services.chatgpt_parser import parse_chatgpt_link, parse_pasted_text
from ..services.translator import translate_to_english
from ..services.trend import classify_trends

router = APIRouter()


class AnalyzeRequest(BaseModel):
    mode: str  # "link" or "paste"
    url: str | None = None
    text: str | None = None


class TurnScore(BaseModel):
    turn_number: int
    human_input_preview: str
    scores: dict[str, float]


class AnalyzeResponse(BaseModel):
    turns: list[TurnScore]
    trends: dict
    summary: dict


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    inference = request.app.state.inference

    if req.mode == "link":
        if not req.url:
            raise HTTPException(400, "url is required for link mode")
        try:
            raw_turns = await parse_chatgpt_link(req.url)
        except Exception as e:
            raise HTTPException(422, f"Failed to parse ChatGPT link: {e}")
    elif req.mode == "paste":
        if not req.text:
            raise HTTPException(400, "text is required for paste mode")
        raw_turns = parse_pasted_text(req.text)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}")

    user_turns = [t["text"] for t in raw_turns if t["role"] == "user"]
    if not user_turns:
        raise HTTPException(422, "No user turns found")

    # Translate non-English to English for the model
    original_turns = user_turns[:]
    user_turns, was_translated = translate_to_english(user_turns)

    turn_scores = inference.predict(user_turns)

    # Restore original text in previews
    if was_translated:
        for i, ts in enumerate(turn_scores):
            if i < len(original_turns):
                ts["human_input_preview"] = original_turns[i][:100]

    trends = classify_trends(turn_scores)

    da_values = [t["scores"]["da_derived"] for t in turn_scores]
    max_da_idx = int(max(range(len(da_values)), key=lambda i: da_values[i]))

    summary = {
        "turn_count": len(turn_scores),
        "avg_da": round(sum(da_values) / len(da_values), 2),
        "max_da_turn": max_da_idx,
        "max_da_value": round(da_values[max_da_idx], 2),
        "overall_trend": trends["score"]["overall"],
        "translated": was_translated,
    }

    return AnalyzeResponse(turns=turn_scores, trends=trends, summary=summary)
