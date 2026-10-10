# WasteValue AI

ถ่ายรูปขยะ รู้ทันทีว่าคืออะไร ทิ้งถังไหน ขายได้ไหม และควรจัดการอย่างไร
โปรเจกต์วิชาปัญญาประดิษฐ์ — จำแนกภาพขยะ 13 คลาสด้วย CNN (MobileNetV2, transfer learning) แล้วแปลงเป็นประเภทขยะ 5 ประเภท

## รันบนเครื่อง
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server.main:app --port 8000
```
เปิด http://localhost:8000
ยังไม่มี `models/model.pt` → ระบบใช้โหมดทดสอบ (สุ่มคำตอบ) อัตโนมัติ

## โครงสร้าง
| โฟลเดอร์ | |
|---|---|
| `notebooks/` | เตรียมข้อมูล (ปาล์ม) และเทรนโมเดล (แฟนต้า) รันบน Google Colab |
| `models/` | `model.pt` จาก notebook เทรน |
| `server/` | FastAPI รับรูป ทำนาย ส่งผลพร้อมข้อมูลจาก `data/` |
| `web/` | หน้าเว็บ |
| `data/` | ข้อมูลรายคลาสและรายประเภทขยะ (ชื่อ ราคา คำแนะนำ) |
| `design/` | ดีไซน์ต้นฉบับ (อ้างอิง) |
| `docs/` | สเปก: SPEC, API, DATA, DESIGN, PLAN |

## ทีม
ปาล์ม — เตรียมข้อมูล · แฟนต้า — โมเดลและประเมินผล · พอร์ต — เว็บและ server

## ข้อมูล
Garbage Classification (12 classes) จาก Kaggle — ปรับเป็น 13 คลาสโดยแบ่ง trash เป็นขยะทั่วไปกับขยะติดเชื้อ
ราคาเป็นค่าประมาณจากแหล่งที่ระบุในหน้าเว็บ ไม่ใช่ราคารับซื้อที่รับประกัน
