from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    inference = getattr(request.app.state, "inference", None)
    return {
        "status": "ready" if inference else "loading",
        "model_loaded": inference is not None,
        "training_turns": inference.n_training_turns if inference else 0,
    }
