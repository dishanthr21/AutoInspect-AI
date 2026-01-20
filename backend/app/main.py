from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes import compare
import os

app = FastAPI(title="Visual QC AI")

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
os.makedirs("uploads/master", exist_ok=True)
os.makedirs("uploads/test", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Connect the API route
app.include_router(compare.router)

# Serve the 'outputs' folder (for result images)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Serve the 'static' folder (for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- THIS IS THE PART THAT WAS MISSING OR NOT SAVED ---
@app.get("/")
async def read_index():
    # This tells the server: "When user visits homepage, give them the HTML file"
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)