# API — ข้อตกลงระหว่าง server กับหน้าเว็บ

## `GET /`
ส่ง `web/index.html`

## `GET /api/health`
```json
{ "ok": true, "fake_model": false, "classes": 13, "threshold": 0.7 }
```

## `POST /api/predict`
- Request: `multipart/form-data` ช่อง `file` (JPG/PNG ≤ 10 MB)
- Response `200`:

```json
{
  "class": "plastic",
  "label_th": "พลาสติก",
  "confidence": 0.93,
  "uncertain": false,
  "runner_up": { "class": "white-glass", "label_th": "แก้วใส", "confidence": 0.04 },
  "category": "recyclable",
  "category_th": "ขยะรีไซเคิล",
  "bin": "ถังขยะรีไซเคิล (สีเหลือง)",
  "material": "พลาสติก",
  "sellable": true,
  "price_min": null,
  "price_max": null,
  "price_unit": "บาท/กก.",
  "price_source": null,
  "price_date": null,
  "prep_steps": ["เทของเหลวออกให้หมด", "ล้างให้สะอาด", "ตากให้แห้ง"],
  "disposal": "ถ้าไม่ขาย ทิ้งในถังขยะรีไซเคิล",
  "warning": null,
  "fake_model": false
}
```

### กฎของแต่ละช่อง
- `class` หนึ่งใน 13 คลาส (`docs/DATA.md`) — มาจาก `class_order` ในไฟล์โมเดล
- `confidence` 0–1 ของคลาสอันดับ 1, `runner_up` คืออันดับ 2 (ส่งเสมอ)
- `uncertain` = `confidence < UNCERTAIN_THRESHOLD` เมื่อเป็น `true` ช่องอื่นยังส่งครบ แต่หน้าเว็บแสดงสถานะ "ไม่แน่ใจ"
- `category`, `category_th`, `bin`, `warning` มาจาก `data/categories.json` ตาม `models/category_map.json`
- `label_th`, `material`, `sellable`, `price_*`, `prep_steps`, `disposal` มาจาก `data/waste_info.json`
- `price_min` / `price_max` เป็น `null` ได้ (ยังไม่มีข้อมูล หรือขายไม่ได้) หน้าเว็บต้องรองรับ
- `fake_model: true` เมื่อรันโหมดสุ่ม หน้าเว็บแสดงป้ายเล็กๆ ว่า "โหมดทดสอบ"

### Error
| status | เมื่อไหร่ | body |
|---|---|---|
| 400 | ไม่มีไฟล์ / ไม่ใช่ภาพ / เปิดภาพไม่ได้ | `{"error": "ข้อความภาษาไทยบอกวิธีแก้"}` |
| 413 | ใหญ่เกิน 10 MB | `{"error": "..."}` |
| 503 | โหลดโมเดลไม่ได้ | `{"error": "..."}` |
