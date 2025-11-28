import os
import json
import shutil, os, uuid
import aiofiles

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
from pipeline.apis import run_search
from db.mongo_client import MongoDBHandler
from db.init_data import seed_products, seed_product_vectors_aligned
from utils.logging_utils import with_logging

# -------------------------------------------------------------------
# Environment settings for stability (macOS M1/M2 + CPU)
# -------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def read_homepage():
    homepage_path = os.path.join("frontend", "index.html")
    not_found_path = os.path.join("frontend", "not_found.html")

    if os.path.exists(homepage_path):
        return FileResponse(homepage_path)
    elif os.path.exists(not_found_path):
        return FileResponse(not_found_path, status_code=404)
    else:
        return HTMLResponse("<h1>404 Not Found</h1>", status_code=404)
    
# -------------------------------------------------------------------
# Global 404 handler
# -------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    """Catch 404 errors and show custom not_found.html if available"""
    if exc.status_code == 404:
        not_found_path = os.path.join("frontend", "not_found.html")
        if os.path.exists(not_found_path):
            return FileResponse(not_found_path, status_code=404)
        else:
            return HTMLResponse("<h1>Oops! Page not found</h1>", status_code=404)
    else:
        # Other errors keep default JSON
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code
        )

# -------------------------------------------------------------------
# Startup Event (seed DB + build index)
# -------------------------------------------------------------------
@app.on_event("startup")
@with_logging("startup_seed_db")
def seed_db():
    mongo = MongoDBHandler()

    # Seed products collection in MongoDB. Drops existing data if force_drop=True.
    seed_products(mongo, force_drop=True)

    # Fetch product IDs for indexing (optionally sampled)
    sample_ids = mongo.get_sample_ids(sample_size=config.SAMPLE_SIZE)
    print(f"Sample Size: {config.SAMPLE_SIZE}")

    # Build Annoy index from combined BERT (text) and DINO (image) embeddings
    # Returns: AnnoyIndex, embedding dimension (dim), and id_map
    index, dim, id_map = seed_product_vectors_aligned(mongo, sample_ids=sample_ids)

    # Save index and metadata
    index.save(config.INDEX_PATH)
    with open(config.ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"id_map": id_map, "dim": dim}, f)

    print(f"Seeding finished.\nIndex saved at {config.INDEX_PATH}\nid_map and dim saved at {config.ID_MAP_PATH} (dim={dim})")

@app.post("/search")
@with_logging("search_request")
async def search(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    query_text: str = Form(""),
):
    """
    At least one of `file` or `image_url` must be provided.
    """
    # Decide embedder mode based on config
    embedder = "local" if config.DEV_MODE else "triton"

    final_url = None
    tmp_upload_dir = config.IMG_UPLOAD_DIR
    os.makedirs(tmp_upload_dir, exist_ok=True)

    # Allow users to upload images from their device,
    # so this is how the uploaded file is handled: save locally and set final_url to local path
    if file:
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(tmp_upload_dir, filename)
 
        # Read and Writethe file content asynchronously using aiofiles
        file_content = await file.read()
        async with aiofiles.open(filepath, "wb") as buffer:
            await buffer.write(file_content)

    # If users use the url for searching an image, use it directly
    # Things can be enhanced: check if the url is safe
    elif image_url:
        final_url = image_url

    else:
        return JSONResponse({"error": "Please provide either an image file or image_url"}, status_code=400)

    # Run the search pipeline
    results = await run_search(
        query_text=query_text,
        query_image_url=final_url,
        embedder_type=embedder
    )

    return results