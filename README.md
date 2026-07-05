# 🎤 Persian Spoken Digit Recognition (PSDR)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-orange)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> تشخیص اعداد گفتاری فارسی (۰ تا ۹) با استفاده از **MFCC** و **LSTM**

---

## 📖 معرفی پروژه

این پروژه یک سیستم **تشخیص اعداد گفتاری فارسی** است که با استفاده از کتابخانه‌های `Librosa` و `PyTorch` پیاده‌سازی شده است.  
مدل با استخراج ویژگی‌های **MFCC** از فایل‌های صوتی و آموزش یک شبکه **LSTM**، قادر به تشخیص رقم گفته‌شده (۰ تا ۹) با دقت **~۸۰٪** است.

### ✨ ویژگی‌ها
- 🔍 استخراج ویژگی‌های MFCC + دلتا + دلتا-دلتا
- 🧠 مدل LSTM دوطرفه با قابلیت تنظیم
- 📊 پشتیبانی از Data Augmentation برای بهبود دقت
- 💾 ذخیره و بارگذاری مدل آموزش‌دیده
- 🎤 پیش‌بینی روی فایل‌های صوتی جدید

---

## 📂 ساختار پروژه
voiceRecognization/
├── voice_recognition.py # استخراج ویژگی‌های MFCC
├── train_model.py # آموزش مدل LSTM
├── predict.py # پیش‌بینی روی فایل جدید
├── processed_data.pkl # داده‌های پردازش‌شده
├── best_digit_model.pth # وزن‌های بهترین مدل
├── requirements.txt # لیست کتابخانه‌های مورد نیاز
└── PSDR/ # دیتاست اصلی


---

## 🚀 نصب و راه‌اندازی

### ۱. کلون کردن پروژه
```bash
git clone https://github.com/your-username/voice-recognition-persian.git
cd voice-recognition-persian

۲. ایجاد محیط مجازی
bash

python3 -m venv voice_env
source voice_env/bin/activate  # برای لینوکس/مک
# voice_env\Scripts\activate   # برای ویندوز

text


---

## 🚀 نصب و راه‌اندازی

### ۱. کلون کردن پروژه
```bash
git clone https://github.com/your-username/voice-recognition-persian.git
cd voice-recognition-persian

۲. ایجاد محیط مجازی
bash

python3 -m venv voice_env
source voice_env/bin/activate  # برای لینوکس/مک
# voice_env\Scripts\activate   # برای ویندوز

۳. نصب کتابخانه‌ها
bash

pip install -r requirements.txt

۴. استخراج ویژگی‌ها از دیتاست
bash

python voice_recognition.py

۵. آموزش مدل
bash

python train_model.py

۶. پیش‌بینی روی یک فایل جدید
bash

python predict.py

 معماری مدل
text

ورودی (MFCC)
    ↓
[LSTM × 2] (Dropout = 0.2)
    ↓
Fully Connected (64 → 10)
    ↓
خروجی (۰ تا ۹)

پارامترها:
بخش	مقدار
تعداد ویژگی‌های ورودی	۳۹ (MFCC + Δ + Δ²)
تعداد لایه‌های LSTM	۲
تعداد نرون‌های پنهان	۱۲۸
نرخ یادگیری	۰.۰۰۰۵
بهینه‌ساز	AdamW
تعداد دوره‌ها	۱۰۰
نتایج
معیار	مقدار
دقت روی داده تست	~۸۰٪
تعداد کل نمونه‌ها	۸۹۷
تعداد کلاس‌ها	۱۰ (۰ تا ۹)
نمودار دقت در طول آموزش
text

دقت (%)
100 ┤
 80 ┤        ╭────╮
 60 ┤   ╭────╯    ╰──╮
 40 ┤ ╭─╯             ╰╮
 20 ┤╭╯                ╰╮
  0 ┼─────────────────────
     0   20   40   60   80  ← دوره


 نمونه‌های اجرا
آموزش مدل
bash

$ python train_model.py
total samples: 897
max length for padding: 86
padded shape: (897, 86, 39)


Epoch 5: Loss = 1.8424, Test Accuracy = 35.00%
Epoch 15: Loss = 1.0410, Test Accuracy = 60.56%
Epoch 35: Loss = 0.5698, Test Accuracy = 71.67%
Epoch 55: Loss = 0.3899, Test Accuracy = 79.44%

 دقت نهایی روی داده تست: 79.44%

```markdown
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10%2B-orange)](https://librosa.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/your-username)

