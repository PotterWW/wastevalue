# PLAN — ลำดับงาน

ทำทีละขั้น แต่ละขั้นต้องผ่าน "เสร็จเมื่อ" ก่อนไปขั้นต่อไป

## ใครทำอะไร (ทั้งทีม)
| คน | งาน | ส่งอะไรเข้า repo |
|---|---|---|
| ปาล์ม | clean ข้อมูล (`notebooks/palm_data_cleaning.ipynb`) | ตัวเลข/กราฟสำหรับสไลด์ข้อ 2 |
| แฟนต้า | เทรน + ประเมินผล (`notebooks/fanta_training.ipynb`) | `models/model.pt`, `confidence_threshold.csv`, กราฟสำหรับสไลด์ข้อ 3–4 |
| A | เว็บ + server + ตารางข้อมูล (repo นี้) | ทุกอย่างใน `server/`, `web/`, `data/` |

## ขั้นที่ 1: Server โหมด fake
- `requirements.txt` ใช้ได้, `server/main.py`, `server/model.py`
- โหมด fake (`FAKE_MODEL=1` หรือไม่มี `models/model.pt`): ใช้ 13 คลาสและ category map จาก `docs/DATA.md` สุ่มความน่าจะเป็น
- `POST /api/predict` ส่ง JSON ตาม `docs/API.md` ครบทุกช่อง, `GET /api/health`
- validate ไฟล์: ชนิด ขนาด เปิดได้
- **เสร็จเมื่อ:** `pytest` ผ่าน: ทุกช่องใน response มีครบ, 13 คลาสครบใน `waste_info.json`, error 400/413 ถูกต้อง

## ขั้นที่ 2: หน้าเว็บ
- `web/index.html` แปลงจาก `design/Main.dc.html` ตาม `docs/DESIGN.md`
- อัปโหลด → `fetch('/api/predict')` → แสดงผลตาม 6 สถานะ → ช่องน้ำหนักคำนวณมูลค่า
- โหมด `?demo` สำหรับสำรอง
- **เสร็จเมื่อ:** เปิดกับ server โหมด fake แล้วอัปโหลดซ้ำจนเห็นครบทั้ง 6 สถานะ, ใช้ได้ที่ความกว้าง 375px, ไม่มี error ใน console

## ขั้นที่ 3: ใส่โมเดลจริง
- วาง `model.pt` จากแฟนต้าที่ `models/model.pt`
- `server/model.py` อ่าน `classes`, `category_map`, `img_size`, `mean`, `std` จากไฟล์ (ตาม `docs/DATA.md`)
- ตั้ง `UNCERTAIN_THRESHOLD` จาก `confidence_threshold.csv`
- **เสร็จเมื่อ:** ทดสอบกับรูปขยะจริงที่ทีมถ่าย 20–30 รูป ผลตรงกับที่ notebook ทายรูปเดียวกัน (ขั้นที่ 8 ใน notebook ของแฟนต้า)

## ขั้นที่ 4: ข้อมูลราคาและตรวจทาน
- ทีมกรอก `price_min/max/source/date` ใน `data/waste_info.json` จากแหล่งจริง
- ตรวจคำแนะนำทุกคลาส และสีถังใน `data/categories.json`
- **เสร็จเมื่อ:** ไม่มีข้อความที่สัญญาเกินโมเดล, ราคาทุกตัวมีแหล่งที่มา

## ขั้นที่ 5: เตรียมวันนำเสนอ
- README มีวิธีรันบนเครื่องใหม่ภายใน 5 นาที
- ซ้อมรันบนโน้ตบุ๊กที่จะใช้จริง ปิด Wi-Fi แล้วลองด้วย
- เตรียมรูปสาธิต: ทายถูก 3–4 รูป (แต่ละประเภท) + รูปที่ทำให้ขึ้น "ไม่แน่ใจ" 1 รูป
