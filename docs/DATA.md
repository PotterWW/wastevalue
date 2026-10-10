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

**แหล่งความจริงคือ `model.pt`** — server ต้องอ่าน `classes` และ `category_map` จากไฟล์นี้ ตารางข้างบนเป็นเอกสารอ้างอิง

## รูปแบบ `models/model.pt` (จาก `notebooks/fanta_training.ipynb`)
`torch.load(path, map_location="cpu", weights_only=False)` ได้ dict:
| key | ค่า |
|---|---|
| `arch` | `"mobilenet_v2"` |
| `state_dict` | น้ำหนักของ `torchvision.models.mobilenet_v2(weights=None)` ที่เปลี่ยน `classifier[1]` เป็น `nn.Linear(1280, len(classes))` |
| `classes` | list 13 ชื่อ เรียงตาม label index |
| `category_map` | dict class → category |
| `img_size` | `224` |
| `mean`, `std` | ค่า normalize |
| `experiment` | การทดลองที่ชนะ (`A_no_aug` หรือ `B_aug`) |

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
