# src/models/hand_models.py
from typing import List, Optional
from pydantic import BaseModel

class Action(BaseModel):
    position: str
    action: str
    amount: Optional[float] = None
    cards: Optional[str] = None

class Hand(BaseModel):
    hero_position: str
    effective_stack: float
    actions: List[Action]
    flop: Optional[str] = None
    turn: Optional[str] = None
    river: Optional[str] = None
    pot_size: Optional[float] = None
