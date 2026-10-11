# DATA — คลาส ประเภทขยะ และไฟล์ข้อมูล

## ชุดข้อมูล
Garbage Classification (12 classes) จาก Kaggle (ผู้อัปโหลด mostafaabla) — ปาล์ม clean ด้วย `notebooks/palm_data_cleaning.ipynb`
- ตัดภาพเสีย ภาพซ้ำ (MD5 + difference hash) ภาพ label ผิดจากการสุ่มตรวจ
- แบ่งโฟลเดอร์ `trash` เป็น 2 คลาสด้วยการคัดด้วยมือ
- จำกัดไม่เกิน 1,000 ภาพต่อคลาส (clothes มีมากเกิน)
- test 15% (stratified) + train 85% แบ่ง 5 fold

## 13 คลาส (ลำดับนี้คือ label index ของโมเดล)
| idx | class | ภาษาไทย | category |
|---|---|---|---|
| 0 | battery | แบตเตอรี่ / ถ่านไฟฉาย | hazardous |
| 1 | biological | ขยะอินทรีย์ / เศษอาหาร | organic |
| 2 | brown-glass | แก้วสีชา | recyclable |
| 3 | cardboard | กล่องกระดาษ | recyclable |
| 4 | clothes | เสื้อผ้า | general |
| 5 | green-glass | แก้วสีเขียว | recyclable |
| 6 | metal | โลหะ / กระป๋อง | recyclable |
| 7 | paper | กระดาษ | recyclable |
| 8 | plastic | พลาสติก | recyclable |
| 9 | shoes | รองเท้า | general |
| 10 | trash_general | ขยะทั่วไป (เช่น แปรงสีฟัน) | general |
| 11 | trash_infectious | ขยะติดเชื้อ (หน้ากากอนามัย ผ้าอนามัย ผ้าอ้อม) | infectious |
| 12 | white-glass | แก้วใส | recyclable |

**แหล่งความจริงของลำดับคลาสคือ `class_order` ในไฟล์โมเดล** — server ต้องอ่านจากไฟล์ ห้าม hardcode ตารางข้างบนเป็นเอกสารอ้างอิง

## รูปแบบไฟล์โมเดล (จากทีมเทรน ไม่ได้เทรนใน repo นี้)
วางที่ `models/` เช่น `models/convnext_tiny_fold1.pt` (ประมาณ 111 MB **ห้าม commit** มี `.gitignore` กันไว้) คู่กับ `models/category_map.json` (commit ได้ อย่าแก้โดยไม่คุยกับทีมเทรน)

`torch.load(path, map_location="cpu")` ได้ dict:
| key | ค่า |
|---|---|
| `model_name` | `"convnext_tiny"` (รองรับ `efficientnet_b0`, `resnet50` ด้วย) |
| `state_dict` | น้ำหนักของ `torchvision.models.convnext_tiny(weights=None)` ที่เปลี่ยน `classifier[2]` เป็น `nn.Linear(in_features, len(class_order))` |
| `class_order` | list 13 ชื่อ เรียงตาม label index |

`models/category_map.json` คือ dict class → category (13 คลาสต้องมีครบ) ค่า mean/std/ขนาดรูปไม่อยู่ในไฟล์ ใช้ค่าคงที่ใน `server/model.py` (224×224, ImageNet mean/std) ซึ่งต้องตรงกับตอนเทรน

## `data/categories.json`
ข้อมูลรายประเภทขยะ: ชื่อไทย ถังที่ทิ้ง ข้อความเตือน (ถ้ามี) และตัวอย่างสำหรับหน้า Waste Guide

> ⚠️ สีถังอิงตามที่ใช้กันทั่วไปในไทย (เหลือง รีไซเคิล / เขียว อินทรีย์ / น้ำเงิน ทั่วไป / แดง อันตราย) **ทีมต้องตรวจกับแหล่งอ้างอิงอีกครั้ง** ขยะติดเชื้อจากบ้านเรือนไม่มีสีถังมาตรฐาน จึงแนะนำให้ใส่ถุงแยกพร้อมเขียนป้าย

## `data/waste_info.json`
ข้อมูลรายคลาส key = ชื่อคลาส
| ช่อง | ความหมาย |
|---|---|
| `label_th` | ชื่อที่แสดง |
| `material` | วัสดุ (`null` ถ้าไม่ใช่ขยะรีไซเคิล) |
| `sellable` | ขายได้ไหม |
| `price_min`, `price_max` | ราคาต่ำ–สูง บาท/กก. **ให้ทีมกรอกจากแหล่งจริง** ตอนนี้เป็น `null` |
| `price_source`, `price_date` | แหล่งและวันที่ของราคา |
| `prep_steps` | คำแนะนำก่อนขาย/ก่อนทิ้ง (ร่างแล้ว ทีมตรวจได้) |
| `disposal` | ถ้าไม่ขาย ทิ้งที่ไหน |
