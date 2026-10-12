# WasteValue AI

ถ่ายรูปขยะ รู้ทันทีว่าคืออะไร ทิ้งถังไหน ขายได้ไหม และควรจัดการอย่างไร
โปรเจกต์วิชาปัญญาประดิษฐ์ — จำแนกภาพขยะ 13 คลาสด้วย CNN (ConvNeXt-Tiny, transfer learning) แล้วแปลงเป็นประเภทขยะ 5 ประเภท

## รันบนเครื่อง (ประมาณ 5 นาที)
ต้องมี Python 3.10 ขึ้นไป เปิด cmd หรือ PowerShell ในโฟลเดอร์ของโปรเจกต์

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn server.main:app --port 8000
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn server.main:app --port 8000
```

เปิด http://localhost:8000 (เครื่องที่ไม่มี GPU ใช้ torch เวอร์ชัน CPU ตามบรรทัดที่ 3 ไฟล์เล็กกว่ามาก ข้ามได้ถ้าอยากลงเวอร์ชันปกติ)
ครั้งต่อไปแค่ activate venv แล้วรัน `uvicorn` อีกครั้ง

### ใส่โมเดลจริง
ไฟล์โมเดล (`convnext_tiny_fold2.pt` ประมาณ 111 MB) **ไม่อยู่ใน git** ขอจากแฟนต้าแล้ววางไว้ที่ `models/` ข้างๆ `category_map.json` แล้วรัน server ใหม่
- ไม่มีไฟล์ `.pt` → ระบบเข้าโหมดทดสอบ (สุ่มคำตอบ) เอง หน้าเว็บจะมีป้าย "โหมดทดสอบ"
- เช็กว่าโหลดโมเดลจริงแล้ว: เปิด http://localhost:8000/api/health ต้องเห็น `"fake_model": false`
- ถ้าเปิดหน้าเว็บแล้วขึ้น "โหมดสาธิต" แสดงว่าหน้าเว็บต่อ server ไม่ได้ (ลอง http://localhost:8000/api/health ดู)

### ทดสอบ
```
pytest -q
```

### ปัญหาที่พบบ่อย
| อาการ | วิธีแก้ |
|---|---|
| `No module named 'torch'` | ยังไม่ได้ activate venv หรือยังไม่ได้ลง torch ตามขั้นตอนข้างบน |
| ขึ้น `[warn] ...` ตอนเปิด server | อ่านข้อความต่อท้าย เช่น โหลดไฟล์โมเดลไม่ได้ หรือ `category_map.json` ไม่ครบ |
| port 8000 ถูกใช้อยู่ | ใช้ `--port 8001` แล้วเปิด http://localhost:8001 |
| ภาษาไทยเพี้ยนในหน้าต่าง cmd | ไม่กระทบการทำงาน ดูผลที่หน้าเว็บ |

### วันนำเสนอ (offline)
- ไม่ต้องใช้อินเทอร์เน็ตตอนรัน (ฟอนต์ Google Fonts มี fallback) ติดตั้งทุกอย่างล่วงหน้าบนเครื่องที่จะใช้ แล้วลองรันโดยปิด Wi-Fi ก่อนวันจริง
- เปิดหน้าเว็บด้วย `?demo` ต่อท้าย (http://localhost:8000/?demo) จะมีปุ่มตัวอย่างผลลัพธ์ 6 แบบ ไม่เรียก server ใช้สำรองได้ถ้าโมเดลมีปัญหา
- กากบาทที่มุมรูปใช้ล้างรูปและผลลัพธ์ก่อนให้คนถัดไปใช้

## โครงสร้าง
| โฟลเดอร์ | |
|---|---|
| `notebooks/` | เตรียมข้อมูล (ปาล์ม) และเทรนโมเดล (แฟนต้า) รันบน Google Colab |
| `models/` | ไฟล์โมเดล `.pt` (ไม่อยู่ใน git วางเอง) และ `category_map.json` |
| `server/` | FastAPI รับรูป ทำนาย ส่งผลพร้อมข้อมูลจาก `data/` |
| `web/` | หน้าเว็บ |
| `data/` | ข้อมูลรายคลาสและรายประเภทขยะ (ชื่อ ราคา คำแนะนำ) |
| `design/` | ดีไซน์ต้นฉบับ (อ้างอิง) |
| `docs/` | สเปก: SPEC, API, DATA, DESIGN, PLAN |

## ทีม
ปาล์ม — เตรียมข้อมูล · แฟนต้า — โมเดลและประเมินผล · พอร์ต — เว็บและ server

## ข้อมูล
Garbage Classification (12 classes) จาก Kaggle — ปรับเป็น 13 คลาสโดยแบ่ง trash เป็นขยะทั่วไปกับขยะติดเชื้อ
ราคาเป็นค่าประมาณจากแหล่งที่ระบุในหน้าเว็บ (วงษ์พาณิชย์ เซนเตอร์ จำกัด ดูเมื่อ 12/10/2026) ราคาเปลี่ยนตามวัน ไม่ใช่ราคารับซื้อที่รับประกัน ที่มาของตัวเลขอยู่ที่ `docs/DATA.md`
