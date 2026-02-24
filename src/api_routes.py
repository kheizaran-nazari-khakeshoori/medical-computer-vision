"""Refactoring api routes for clarity."""
from fastapi import APIRouter
router = APIRouter()
@router.get("/health")
def health(): return {"status": "ok"}
@router.get("/models")
def list_models_route():
    from src.registry import list_models
    return list_models()
