# WasteValue AI

โปรเจกต์วิชา AI (กลุ่ม 3 คน) เว็บที่รับรูปขยะ 1 ชิ้น แล้วบอกว่าเป็นอะไร ทิ้งถังไหน ขายได้ไหม ราคาประมาณเท่าไหร่ และควรเตรียมอย่างไร
โมเดลเป็น CNN (MobileNetV2 transfer learning) ทายได้ 13 คลาส ส่วนประเภทขยะ ราคา และคำแนะนำมาจากตารางใน `data/`

## Stack
- Backend: Python 3.10+, FastAPI, PyTorch/torchvision (CPU พอ) — `server/`
- Frontend: HTML + CSS + JavaScript ล้วน ไฟล์เดียว ไม่ใช้ framework ไม่มี build step — `web/index.html`
- โมเดลเทรนบน Google Colab ด้วย `notebooks/` ไม่ได้เทรนใน repo นี้

## คำสั่ง
```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 8000     # เปิด http://localhost:8000
FAKE_MODEL=1 uvicorn server.main:app --reload    # โหมดสุ่มคำตอบ ใช้ระหว่างรอ model.pt
pytest -q                                        # ทดสอบ API
```

## โครงสร้าง
- `server/main.py` FastAPI: เสิร์ฟ `web/index.html` ที่ `/` และ `POST /api/predict`
- `server/model.py` โหลด `models/model.pt` และทำนาย (มีโหมด fake)
- `web/index.html` หน้าเว็บทั้งหมด **ทำเสร็จแล้ว** (แปลงจาก `design/Main.dc.html`) แก้เฉพาะจุด ห้ามเขียนใหม่ทั้งไฟล์
- `data/waste_info.json` ข้อมูลรายคลาส (ชื่อไทย วัสดุ ราคา คำแนะนำ) / `data/categories.json` ข้อมูลรายประเภทขยะ
- `docs/` สเปกทั้งหมด อ่านก่อนเริ่มงาน: SPEC → API → DATA → DESIGN → PLAN

## กฎที่ห้ามพลาด
- **ลำดับคลาสต้องตรงกับ `model.pt`** อ่านรายชื่อคลาสจาก `model.pt` เสมอ ห้าม hardcode ลำดับเอง (ดู `docs/DATA.md`)
- **Preprocess ต้องตรงกับตอนเทรน:** RGB → resize เป็น 224×224 ตรงๆ (ไม่ crop) → ToTensor → Normalize mean `[0.485,0.456,0.406]` std `[0.229,0.224,0.225]`
- **ห้ามแต่งตัวเลขราคาหรือแหล่งที่มา** ราคาอยู่ใน `data/waste_info.json` เท่านั้น ถ้าเป็น `null` ให้หน้าเว็บแสดงว่ายังไม่มีข้อมูลราคา
- **ข้อความบนเว็บต้องไม่สัญญาเกินที่โมเดลทำได้** เช่น ห้ามเขียน "PET" หรือชื่อชนิดขยะที่ไม่มีใน 13 คลาส
- **ข้อมูลประเภทขยะมี 2 ที่ต้องตรงกัน:** `data/categories.json` (server ใช้) กับ `CATEGORY_UI` ใน `web/index.html` (ใช้กับ Waste Guide และสีถัง) แก้ที่หนึ่งต้องแก้อีกที่ด้วย
- ตัวอย่างใน `DEMO` ของ `web/index.html` ต้องตรงกับ `data/waste_info.json` (ราคาตัวอย่างพลาสติก 8–12 เป็นค่าสมมติจากดีไซน์ เปลี่ยนเมื่อได้ราคาจริง)
- หน้าเว็บต้องไม่พึ่ง CDN ที่ไม่จำเป็น ใช้ Google Fonts ได้ (มี fallback) เพราะวันนำเสนออาจรันแบบ offline บนโน้ตบุ๊ก
- ไม่ต้อง deploy ขึ้นอินเทอร์เน็ต รันบน localhost ตอนนำเสนอ
- UI เป็นภาษาไทย โค้ดและชื่อตัวแปรเป็นภาษาอังกฤษ

## วิธีทำงาน
- ทำตามลำดับใน `docs/PLAN.md` ทีละขั้น ทดสอบให้ผ่านก่อนขึ้นขั้นถัดไป
- ถ้าสเปกขัดกันหรือไม่ชัด ให้ถามก่อน อย่าเดา
- `notebooks/` เป็นของเพื่อนในทีม อย่าแก้ ถ้าจำเป็นต้องแก้ให้บอกก่อน