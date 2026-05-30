from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid

# Import the core PDF generation function from your extracted script
from FCN_core import build_fcn_pdf

app = FastAPI(title="FCN PDF Generator API")

# Define the expected JSON payload from n8n based on your script's parameters
class FCNRequest(BaseModel):
    tickers: str
    tenor: int = 6
    strike: float = None
    ko: float = None
    ki: float = None
    currency: str = "USD"
    coupon: float = None
    ko_type: str = "Daily Close"
    ki_type: str = "At Maturity"

def remove_file(path: str):
    """Deletes the PDF from the server after sending it to n8n to save space"""
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

@app.post("/generate-pdf")
async def create_pdf(req: FCNRequest, background_tasks: BackgroundTasks):
    try:
        # Generate a unique filename to prevent overwriting if multiple requests occur
        unique_id = uuid.uuid4().hex[:8]
        safe_tickers = req.tickers.replace(",", "_").replace(" ", "")
        filename = f"FCN_{safe_tickers}_{unique_id}.pdf"
        
        # Call your core function
        build_fcn_pdf(
            tickers=req.tickers,
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

        # Schedule the file for deletion AFTER it is sent back to n8n
        background_tasks.add_task(remove_file, filename)
        
        # Return the PDF file directly
        return FileResponse(
            path=filename, 
            filename=f"FCN_{safe_tickers}.pdf", 
            media_type='application/pdf'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))