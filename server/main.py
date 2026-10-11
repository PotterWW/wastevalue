import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from .model import ModelUnavailable, get_predictor, load_json

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "web" / "index.html"
MAX_BYTES = 10 * 1024 * 1024
PRICE_UNIT = "บาท/กก."

WASTE_INFO = {k: v for k, v in load_json("waste_info.json").items() if not k.startswith("_")}
CATEGORIES = load_json("categories.json")

@asynccontextmanager
async def lifespan(app):
    # Load the model before serving: the page's health check gives up after 2.5 s
    # and would drop into demo mode if the first request had to load a large file.
    try:
        get_predictor(threshold())
    except ModelUnavailable as e:
        print(f"[warn] {e}")
    yield


app = FastAPI(title="WasteValue AI", lifespan=lifespan)

def threshold():
    try:
        return float(os.environ.get("UNCERTAIN_THRESHOLD", "0.7"))
    except ValueError:
        return 0.7

def error(status, message):
    return JSONResponse({"error": message}, status_code=status)

@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    return error(400, "ส่งข้อมูลไม่ถูกต้อง กรุณาเลือกรูป JPG หรือ PNG แล้วลองใหม่")

@app.get("/")
def index():
    if not INDEX_HTML.exists():
        return error(404, "ไม่พบไฟล์ web/index.html")
    return FileResponse(INDEX_HTML)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)  # no icon; avoids a red 404 in the browser console


@app.get("/api/health")
def health():
    try:
        p = get_predictor(threshold())
        return {"ok": True, "fake_model": p.fake, "classes": len(p.classes), "threshold": threshold()}
    except ModelUnavailable as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

@app.post("/api/predict")
async def predict(file: UploadFile | None = File(None)):
    if file is None or not file.filename:
        return error(400, "ไม่พบไฟล์ กรุณาเลือกรูป JPG หรือ PNG ก่อนกดวิเคราะห์")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return error(413, "ไฟล์ใหญ่เกิน 10 MB กรุณาเลือกรูปที่เล็กลง")
    if not data:
        return error(400, "ไฟล์ว่างเปล่า กรุณาเลือกรูปใหม่")
    try:
        image = Image.open(io.BytesIO(data))
        fmt = image.format
        image.load()
    except Exception:
        return error(400, "เปิดรูปไม่ได้ กรุณาใช้ไฟล์ภาพ JPG หรือ PNG ที่ไม่เสียหาย")
    if fmt not in ("JPEG", "PNG"):
        return error(400, "รองรับเฉพาะไฟล์ JPG และ PNG เท่านั้น")

    thr = threshold()
    try:
        predictor = get_predictor(thr)
        pred = predictor.predict(image)
    except ModelUnavailable as e:
        return error(503, f"ระบบยังไม่พร้อมใช้งาน: {e}")

    info = WASTE_INFO.get(pred.top_class)
    runner_info = WASTE_INFO.get(pred.runner_class)
    cat_key = predictor.category_map.get(pred.top_class)
    cat = CATEGORIES.get(cat_key)
    if info is None or runner_info is None or cat is None:
        return error(503, f"ข้อมูลของคลาส {pred.top_class} ยังไม่ครบใน data/ กรุณาแจ้งทีมพัฒนา")

    sellable = bool(info["sellable"])
    return {
        "class": pred.top_class,
        "label_th": info["label_th"],
        "confidence": round(pred.top_confidence, 4),
        "uncertain": pred.top_confidence < thr,
        "runner_up": {
            "class": pred.runner_class,
            "label_th": runner_info["label_th"],
            "confidence": round(pred.runner_confidence, 4),
        },
        "category": cat_key,
        "category_th": cat["category_th"],
        "bin": cat["bin"],
        "material": info["material"],
        "sellable": sellable,
        "price_min": info["price_min"] if sellable else None,
        "price_max": info["price_max"] if sellable else None,
        "price_unit": PRICE_UNIT,
        "price_source": info["price_source"] if sellable else None,
        "price_date": info["price_date"] if sellable else None,
        "prep_steps": info["prep_steps"],
        "disposal": info["disposal"],
        "warning": cat["warning"],
        "fake_model": predictor.fake,
    }
