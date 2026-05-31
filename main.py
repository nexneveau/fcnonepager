from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid
import requests
from typing import Optional  # <--- NEW IMPORT ADDED HERE

# ==========================================
# YAHOO FINANCE RENDER BLOCK BYPASS
# ==========================================
original_session_request = requests.Session.request
def patched_session_request(self, method, url, **kwargs):
    kwargs.setdefault('headers', {})
    kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    return original_session_request(self, method, url, **kwargs)
requests.Session.request = patched_session_request

original_request = requests.request
def patched_request(method, url, **kwargs):
    kwargs.setdefault('headers', {})
    kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    return original_request(method, url, **kwargs)
requests.request = patched_request
# ==========================================

from FCN_core import build_pdf

app = FastAPI(title="FCN PDF Generator API")

# ==========================================
# FIX: ALLOW NULL VALUES USING 'Optional'
# ==========================================
class FCNRequest(BaseModel):
    tickers: str
    tenor: int = 6
    strike: Optional[float] = None
    ko: Optional[float] = None
    ki: Optional[float] = None
    currency: str = "USD"
    coupon: Optional[float] = None
    ko_type: str = "Daily Close"
    ki_type: str = "At Maturity"

def remove_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

@app.post("/generate-pdf")
async def create_pdf(req: FCNRequest, background_tasks: BackgroundTasks):
    try:
        # Prevent crash if tickers is completely empty
        if not req.tickers or req.tickers.strip() == "":
            raise Exception("No tickers provided.")

        ticker_list = [t.strip() for t in req.tickers.split(",") if t.strip()]
        unique_id = uuid.uuid4().hex[:8]
        safe_name = "_".join(ticker_list)
        filename = f"FCN_{safe_name}_{unique_id}.pdf"
        
        build_pdf(
            tickers=ticker_list,
            tenor=req.tenor,
            strike=req.strike,
            ko=req.ko,
            ki=req.ki,
            currency=req.currency,
            coupon=req.coupon,
            ko_type=req.ko_type,
            ki_type=req.ki_type,
            out_pdf=filename
        )
        
        if not os.path.exists(filename):
            raise Exception("PDF generation failed. File not found.")

        background_tasks.add_task(remove_file, filename)
        
        return FileResponse(
            path=filename, 
            filename=f"FCN_{safe_name}.pdf", 
            media_type='application/pdf'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
