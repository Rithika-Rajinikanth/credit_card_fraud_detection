from pydantic import BaseModel, Field
from typing import Optional

class Transaction(BaseModel):
    V1: Optional[float] = 0
    V2: Optional[float] = 0
    V3: Optional[float] = 0
    V4: Optional[float] = 0
    V5: Optional[float] = 0
    V6: Optional[float] = 0
    V7: Optional[float] = 0
    V8: Optional[float] = 0
    V9: Optional[float] = 0
    V10: Optional[float] = 0
    V11: Optional[float] = 0
    V12: Optional[float] = 0
    V13: Optional[float] = 0
    V14: Optional[float] = 0
    V15: Optional[float] = 0
    V16: Optional[float] = 0
    V17: Optional[float] = 0
    V18: Optional[float] = 0
    V19: Optional[float] = 0
    V20: Optional[float] = 0
    V21: Optional[float] = 0
    V22: Optional[float] = 0
    V23: Optional[float] = 0
    V24: Optional[float] = 0
    V25: Optional[float] = 0
    V26: Optional[float] = 0
    V27: Optional[float] = 0
    V28: Optional[float] = 0

    Amount: float = Field(..., gt=0)
    Hour: int = Field(..., ge=0, le=23)
