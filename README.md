# Human3D-AI

โปรเจกต์ตัวอย่างสำหรับตรวจสอบรูปภาพหลายใบว่าเป็น `human` หรือ `fake` ด้วย PyTorch + OpenCV

## โครงสร้างข้อมูล
จัดชุดข้อมูลเป็นโฟลเดอร์แบบนี้:

```text
dataset/
├── human/
│   ├── sample001/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── 3.jpg
│   └── sample002/
└── fake/
    ├── sample001/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── sample002/
```

- 1 โฟลเดอร์ = 1 ตัวอย่าง
- ในแต่ละโฟลเดอร์มีภาพหลายใบของตัวอย่างนั้น
- โฟลเดอร์ `human` คือภาพคนจริง
- โฟลเดอร์ `fake` คือภาพปลอม เช่น รูปถ่าย, หน้าจอ, หรือ replay

## ติดตั้ง
```bash
pip install -r requirements.txt
```

## เทรน
```bash
python train.py --dataset dataset --epochs 10 --batch-size 4 --max-images 5
```

ไฟล์โมเดลจะถูกบันทึกที่ `model/best.pt`

## ทำนาย
```bash
python predict.py --model model/best.pt --images a.jpg b.jpg c.jpg
```

หรือส่งทั้งโฟลเดอร์
```bash
python predict.py --model model/best.pt --folder path/to/images
```

## API
```bash
python api.py --model model/best.pt
```

แล้วส่ง `POST /predict` แบบ `multipart/form-data` โดยใช้ field ชื่อ `images`

## หมายเหตุ
- ตั้งค่า `--pretrained` ใน `train.py` ถ้าต้องการใช้ backbone ที่มีน้ำหนัก pretrained
- ถ้าไม่มีอินเทอร์เน็ต ให้เทรนแบบไม่ใช้ pretrained ได้
- โปรเจกต์นี้เป็น baseline เริ่มต้น สามารถต่อยอดกับ pattern/depth feature ได้
