# โมเดล EfficientNet-V2-S (Transfer Learning): สถาปัตยกรรมและผลการทดลอง

## 1. ภาพรวมงาน

| หัวข้อ    | รายละเอียด                                                       |
| --------- | ---------------------------------------------------------------- |
| ประเภทงาน | Image Classification (Multi-class)                               |
| จำนวนคลาส | 12 คลาส (ขยะประเภทต่าง ๆ)                                        |
| Input     | ภาพสี RGB 3 channels, Resize เป็น 224 x 224                      |
| Output    | ค่า logits 12 ค่า (ทายคลาสด้วย argmax)                           |
| โมเดล     | EfficientNet-V2-S, Pretrained จาก ImageNet (`weights="DEFAULT"`) |

---

## 2. ภาพรวมโครงสร้าง

```
Input (3 x 224 x 224)
        |
        v
+--------------------------------------------+
|  Backbone: model.features                  |   <- ของ EfficientNet (Pretrained)
|  8 stages, 40 blocks                       |
|  Activation: SiLU, มี BatchNorm            |
+--------------------------------------------+
        |   feature map 1280 x 7 x 7
        v
Global Average Pooling  ->  เวกเตอร์ยาว 1280
        |
        v
+--------------------------------------------+
|  Classifier (หัวที่เปลี่ยนใหม่)            |   <- ส่วนที่ปรับเพิ่ม
|  Dropout(p=0.2)                            |
|  Linear(1280 -> 12)                        |
+--------------------------------------------+
        |
        v
Output: logits 12 ค่า  ->  CrossEntropyLoss
```

---

## 3. รายละเอียด Backbone (`model.features`)

| Stage | ชนิดบล็อก | จำนวนบล็อก | Channels ออก | ขนาด feature map |
|---|---|---|---|---|
| 0 | Conv 3x3 (stem) | 1 | 24 | 112 x 112 |
| 1 | FusedMBConv | 2 | 24 | 112 x 112 |
| 2 | FusedMBConv | 4 | 48 | 56 x 56 |
| 3 | FusedMBConv | 4 | 64 | 28 x 28 |
| 4 | MBConv + SE | 6 | 128 | 14 x 14 |
| 5 | MBConv + SE | 9 | 160 | 14 x 14 |
| 6 | MBConv + SE | 15 | 256 | 7 x 7 |
| 7 | Conv 1x1 | 1 | 1280 | 7 x 7 |

รวม 40 บล็อก (FusedMBConv 10 บล็อก + MBConv 30 บล็อก) นอกนั้นคือ Conv stem และ Conv 1x1 ท้าย

### จำนวนชั้นทั้งหมด (นับจาก `print(model)`)

| ส่วน                             | Conv2d  | หมายเหตุ                                         |
| -------------------------------- | ------- | ------------------------------------------------ |
| Stem (stage 0)                   | 1       | Conv 3x3 stride 2                                |
| FusedMBConv 10 บล็อก (stage 1-3) | 18      | stage 1 บล็อกละ 1 ชั้น, stage 2-3 บล็อกละ 2 ชั้น |
| MBConv 30 บล็อก (stage 4-6)      | 90      | บล็อกละ 3 ชั้น (expand, depthwise, project)      |
| Conv 1x1 ท้าย (stage 7)          | 1       | ออก 1280 channels                                |
| **รวมชั้น Conv หลัก**            | **110** |                                                  |
| Conv ในตัว SE (30 บล็อก x 2)     | 60      | ชั้นเล็กสำหรับให้น้ำหนัก channel                 |
| **รวม Conv2d ทั้งหมด**           | **170** |                                                  |
| Linear (Classifier)              | 1       | 1280 -> 12 ส่วนที่เพิ่มใหม่                      |

### โครงสร้างภายในแต่ละบล็อก

```
FusedMBConv (stage 1-3)
  Conv 3x3 (ขยาย channel) + BatchNorm + SiLU
  -> Conv 1x1 (บีบกลับ) + BatchNorm
  (stage 1 ขยาย x1 มี Conv ตัวเดียว, stage 2-3 ขยาย x4)

MBConv (stage 4-6)
  Conv 1x1 (ขยาย channel) + BatchNorm + SiLU
  -> Depthwise Conv 3x3 + BatchNorm + SiLU
  -> Squeeze-and-Excitation
  -> Conv 1x1 (บีบกลับ) + BatchNorm
  (stage 4 ขยาย x4, stage 5-6 ขยาย x6)
```

- บล็อกที่ขนาด input กับ output เท่ากัน จะบวกค่า input กลับเข้าไป (skip connection)
- stride 2 อยู่ที่ stem และบล็อกแรกของ stage 2, 3, 4, 6 ทำให้ภาพลดจาก 224 เหลือ 7 x 7

### ส่วนประกอบหลัก (อธิบายแบบสั้น)

- **Conv2D + BatchNorm + SiLU:** ชุดพื้นฐานเหมือน Basic CNN ที่เรียนมา แต่ใช้ SiLU แทน ReLU
- **MBConv:** แทนที่จะใช้ Conv ธรรมดา ใช้ 3 ขั้น คือ ขยายจำนวน channel, กรองแยกทีละ channel (Depthwise Conv), แล้วบีบกลับ ทำให้ลึกได้แต่ใช้พารามิเตอร์น้อย
- **FusedMBConv:** รวมสองขั้นแรกของ MBConv เป็น Conv 3x3 ตัวเดียว ใช้ใน stage ต้น ๆ เพราะเร็วกว่าบน GPU
- **Squeeze-and-Excitation (SE):** เลเยอร์เล็ก ๆ ที่ให้น้ำหนักว่า channel ไหนสำคัญ ใช้ใน stage 4 ถึง 6 (ย่อ channel ลงเหลือ 1/4 แล้วขยายกลับ ผ่าน Sigmoid ได้ค่า 0-1 ต่อ channel)
- **Stochastic Depth:** สุ่มข้ามบล็อกตอนเทรน เป็น Regularization ความน่าจะเป็นเพิ่มตามความลึก ตั้งแต่ 0.0 ที่บล็อกแรกถึง 0.195 ที่บล็อกสุดท้าย (ค่า default ของ torchvision)
- **Global Average Pooling:** ยุบ feature map 7 x 7 เหลือค่าเฉลี่ยต่อ channel ได้เวกเตอร์ยาว 1280

---

## 4. ส่วนที่ปรับเพิ่มจาก EfficientNet เดิม (Classifier)

```python
model = models.efficientnet_v2_s(weights="DEFAULT")

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=12)
)
```

- หัวเดิมของ ImageNet ทาย 1000 คลาส เปลี่ยนเป็น 12 คลาส
- ไม่มี Softmax ในโมเดล เพราะ `CrossEntropyLoss` รวม Softmax ไว้ให้แล้ว
- พารามิเตอร์ของหัวใหม่: 1280 x 12 + 12 = **15,372 ตัว**

---

## 5. การตั้งค่าการเทรน

| หัวข้อ | ค่าที่ใช้ |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Activation | SiLU (ใน Backbone), Softmax (รวมใน CrossEntropyLoss) |
| Batch size | 32 |
| LR Scheduler | ไม่มี |
| Regularization | Dropout 0.2 ที่หัว, Stochastic Depth ใน Backbone |
| Augmentation (train) | RandomHorizontalFlip |
| Preprocessing | Resize 224 x 224, ToTensor, Normalize (mean/std ของ ImageNet) |
| Val / Test | Resize 224 x 224, Normalize เท่านั้น (ไม่มี augmentation) |
| ข้อมูล | data/train, data/val, data/test |

---

## 6. สามเวอร์ชันที่ทดลอง (โมเดลเดียวกัน ต่างกันที่วิธีเทรน)

| เวอร์ชัน | Backbone | LR | Epochs | Val Acc |
|---|---|---|---|---|
| v1 Baseline | Freeze ทั้งหมด (เทรนแค่หัว) | 1e-3 | 10 | ~91.5% |
| v2 Unfreeze ทันที | เทรนทั้งโมเดลตั้งแต่ต้น | 1e-4 | 10 | ~88.5% |
| v3 Two-stage | Stage 1: Freeze แล้ว Stage 2: Unfreeze | 1e-3 แล้ว 1e-4 | 5 + 10 | ~97.5% |

ผล v3 บน Test: **Accuracy 97.94%**, Macro F1 0.9773 (v1 และ v2 วัดเฉพาะ Val ไม่ได้วัดบน Test)

จำนวนพารามิเตอร์ที่เทรนจริง:
- v1 และ Stage 1 ของ v3: 15,372 ตัว (เฉพาะหัว)
- v2 และ Stage 2 ของ v3: ประมาณ 20.2M ตัว (ทั้งโมเดล)
