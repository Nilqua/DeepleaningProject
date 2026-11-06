v1: Baseline (Freeze Backbone)
Freeze Backbone ทั้งหมด
เปลี่ยนหัวเป็น Dropout(0.2) + Linear(1280, 12) เทรนเฉพาะหัว
Adam lr 0.001 คงที่, batch 32, 10 epochs, augmentation แค่ RandomHorizontalFlip
ผล: Val Acc ~91.5%, Train ≈ Val, Loss นิ่งที่ ~0.26 จึงสรุปว่า Underfit

              precision    recall  f1-score   support

     battery     0.8718    0.9577    0.9128       142
  biological     0.9533    0.9662    0.9597       148
 brown-glass     0.9176    0.8571    0.8864        91
   cardboard     0.9051    0.9323    0.9185       133
     clothes     0.9735    0.9778    0.9756       225
 green-glass     0.8235    0.9130    0.8660        92
       metal     0.8411    0.7826    0.8108       115
       paper     0.9038    0.8981    0.9010       157
     plastic     0.8385    0.8385    0.8385       130
       shoes     0.9795    0.9696    0.9745       296
       trash     0.9429    0.9429    0.9429       105
 white-glass     0.8835    0.7845    0.8311       116

    accuracy                         0.9154      1750
   macro avg     0.9028    0.9017    0.9015      1750
weighted avg     0.9158    0.9154    0.9150      1750


v2: Full Fine-tuning (Unfreeze ทั้งโมเดล)
โมเดลและหัวเหมือน v1 แต่ปลด Freeze ทุกเลเยอร์ (requires_grad = True)
ลด lr เป็น 0.0001 เพื่อไม่ให้ Pretrained Weights พัง
ส่วนอื่นเท่าเดิม (batch 32, Adam, augmentation, ไม่มี Scheduler) เปลี่ยนแค่ตัวนี้ตัวเดียวเพื่อดูผลของ Unfreeze ล้วน ๆ
เหตุผล: กราฟ v1 แสดง Underfit จึงให้ Backbone ปรับฟีเจอร์เข้ากับรูปขยะ

              precision    recall  f1-score   support

     battery     0.8784    0.9155    0.8966       142
  biological     0.9342    0.9595    0.9467       148
 brown-glass     0.8625    0.7582    0.8070        91
   cardboard     0.8633    0.9023    0.8824       133
     clothes     0.9515    0.9600    0.9558       225
 green-glass     0.8587    0.8587    0.8587        92
       metal     0.7941    0.7043    0.7465       115
       paper     0.8688    0.8854    0.8770       157
     plastic     0.7820    0.8000    0.7909       130
       shoes     0.9572    0.9831    0.9700       296
       trash     0.9158    0.8286    0.8700       105
 white-glass     0.7712    0.7845    0.7778       116

    accuracy                         0.8851      1750
   macro avg     0.8698    0.8617    0.8649      1750
weighted avg     0.8844    0.8851    0.8842      1750

v3: Two-stage Fine-tuning
โมเดลเหมือน v1 และ v2 คือ EfficientNet-V2-S Pretrained ImageNet และหัว Dropout(0.2) + Linear(1280, 12)
Stage 1 (epoch 1-5): Freeze Backbone เทรนเฉพาะหัวด้วย Adam lr 0.001 ให้หัวที่เริ่มจากค่าสุ่มปรับตัวก่อน
Stage 2 (epoch 6-15): Unfreeze ทั้งโมเดล เทรนต่อจากโมเดลเดิมด้วย Adam lr 0.0001 โดยไม่สร้างโมเดลใหม่
ส่วนอื่นเท่าเดิม (batch 32, augmentation แค่ Flip, ไม่มี Scheduler) เปลี่ยนแค่ลำดับการเทรน
เหตุผล: v2 ที่ Unfreeze ทันทีได้ 88.5% ต่ำกว่า v1 เพราะ Gradient จากหัวที่ยังสุ่มอยู่ไปรบกวน Pretrained Weights ของ Backbone จึงให้หัวเรียนรู้ก่อน แล้วค่อยปล่อยให้ Backbone ปรับตัวช้า ๆ

              precision    recall  f1-score   support

     battery     0.9789    0.9789    0.9789       142
  biological     0.9800    0.9932    0.9866       148
 brown-glass     0.9778    0.9670    0.9724        91
   cardboard     0.9846    0.9624    0.9734       133
     clothes     0.9955    0.9867    0.9911       225
 green-glass     0.9674    0.9674    0.9674        92
       metal     0.9231    0.9391    0.9310       115
       paper     0.9625    0.9809    0.9716       157
     plastic     0.9323    0.9538    0.9430       130
       shoes     0.9932    0.9932    0.9932       296
       trash     0.9811    0.9905    0.9858       105
 white-glass     0.9820    0.9397    0.9604       116

    accuracy                         0.9749      1750
   macro avg     0.9715    0.9711    0.9712      1750
weighted avg     0.9751    0.9749    0.9749      1750

Test score (v3) :
              precision    recall  f1-score   support

     battery     0.9859    0.9859    0.9859       142
  biological     0.9669    0.9865    0.9766       148
 brown-glass     0.9778    0.9670    0.9724        91
   cardboard     0.9848    0.9701    0.9774       134
     clothes     0.9955    0.9867    0.9911       225
 green-glass     0.9677    0.9783    0.9730        92
       metal     0.9561    0.9478    0.9520       115
       paper     0.9748    0.9810    0.9779       158
     plastic     0.9474    0.9767    0.9618       129
       shoes     0.9932    0.9865    0.9899       297
       trash     0.9905    1.0000    0.9952       104
 white-glass     0.9825    0.9655    0.9739       116

    accuracy                         0.9794      1751
   macro avg     0.9769    0.9777    0.9773      1751
weighted avg     0.9796    0.9794    0.9795      1751