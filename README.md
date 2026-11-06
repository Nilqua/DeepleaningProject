# Garbage Image Classification (12 คลาส)

Term Project วิชา 517433 Deep Learning for Computer Vision
จำแนกประเภทขยะจากภาพ ด้วย Transfer Learning (PyTorch)

## ภาพรวม

| หัวข้อ | รายละเอียด |
|---|---|
| ประเภทงาน | Image Classification (Multi-class) |
| จำนวนคลาส | 12 คลาส |
| Input | ภาพ RGB 3 channels, Resize 224 x 224 |
| Output | คลาสที่ทำนาย (12 คลาส) |
| โมเดลในส่วนนี้ | EfficientNet-V2-S (Pretrained ImageNet) |

คลาสทั้งหมด: `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`

## ชุดข้อมูล

แบ่งสัดส่วน Train / Val / Test ประมาณ 70 / 15 / 15

| Subset | จำนวนภาพ |
|---|---|
| Train | 8,168 |
| Val | 1,750 |
| Test | 1,751 |

## โครงสร้างโปรเจกต์

```
DLProj/
├── README.md
├── requirements.txt
├── src/                # โค้ดเทรนและ notebook การทดลอง
│   └── exp1_baseline.ipynb
├── notebooks/
└── saves/              # น้ำหนักโมเดลที่บันทึกไว้
```

เปิด `src/exp1_baseline.ipynb`

## โมเดลและการเทรน

EfficientNet-V2-S ที่ผ่านการเทรนจาก ImageNet เปลี่ยนส่วนหัว (Classifier) เป็น `Dropout(0.2)` + `Linear(1280, 12)`

| หัวข้อ | ค่าที่ใช้ |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Batch size | 32 |
| Augmentation (train) | RandomHorizontalFlip |
| Preprocessing | Resize 224 x 224, Normalize (mean/std ของ ImageNet) |

## การทดลองและผลลัพธ์

ทดลอง 3 เวอร์ชัน ปรับทีละอย่างโดยดูปัญหาจากกราฟ Accuracy/Loss

| เวอร์ชัน | วิธีเทรน | LR | Epochs | Val Acc |
|---|---|---|---|---|
| v1 Baseline | Freeze Backbone เทรนเฉพาะหัว | 1e-3 | 10 | ~91.5% |
| v2 Full Fine-tuning | Unfreeze ทั้งโมเดลตั้งแต่ต้น | 1e-4 | 10 | ~88.5% |
| v3 Two-stage | Stage 1 เทรนหัว แล้ว Stage 2 Unfreeze ทั้งโมเดล | 1e-3 แล้ว 1e-4 | 5 + 10 | ~97.5% |

- **v1:** Train และ Val เท่ากันและ Loss นิ่ง จึงสรุปว่า Underfit
- **v2:** Unfreeze ทันทีขณะที่หัวยังเริ่มจากค่าสุ่ม Gradient ไปรบกวน Pretrained Weights ทำให้ผลแย่กว่า v1
- **v3:** ให้หัวเรียนก่อน แล้วค่อย Unfreeze ด้วย LR ต่ำ ผลดีขึ้นชัดเจน

### ผลบน Test Set (v3)

| Metric | ค่า |
|---|---|
| Accuracy | 97.94% |
| Macro F1 | 0.9773 |

คลาสที่ยังอ่อนที่สุดคือ `metal` (F1 0.952) และ `plastic` (F1 0.962) v1 และ v2 วัดเฉพาะ Val ไม่ได้วัดบน Test

## การวัดผล

ฟังก์ชันที่ใช้ใน notebook

- `plot_history(history)` กราฟ Accuracy และ Loss ของ Train เทียบ Validation
- `evaluate(model, loader, class_names)` Classification Report และ Confusion Matrix
- `plot_wrong(dataset, y_true, y_pred, class_names)` แสดงภาพที่ทำนายผิด
