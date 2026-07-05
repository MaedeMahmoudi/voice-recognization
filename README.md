#  Persian Spoken Digit Recognition (PSDR)
data set = https://github.com/Ralireza/PSDR
> تشخیص اعداد گفتاری فارسی (۰ تا ۹) با استفاده از **MFCC** و **LSTM**

---

##معرفی پروژه

این پروژه یک سیستم **تشخیص اعداد گفتاری فارسی** است که با استفاده از کتابخانه‌های `Librosa` و `PyTorch` پیاده‌سازی شده است.  
مدل با استخراج ویژگی‌های **MFCC** از فایل‌های صوتی و آموزش یک شبکه **LSTM**، قادر به تشخیص رقم گفته‌شده (۰ تا ۹) با دقت **~۸۰٪** است.

###ویژگی‌ها
-  استخراج ویژگی‌های MFCC + دلتا + دلتا-دلتا
- مدل LSTM دوطرفه با قابلیت تنظیم
- ذخیره و بارگذاری مدل آموزش‌دیده
- پیش‌بینی روی فایل‌های صوتی جدید

---

## ساختار پروژه
voiceRecognization/
├── voice_recognition.py # استخراج ویژگی‌های MFCC
├── train_model.py # آموزش مدل LSTM
├── predict.py # پیش‌بینی روی فایل جدید
├── processed_data.pkl # داده‌های پردازش‌شده
├── best_digit_model.pth # وزن‌های بهترین مدل
├── requirements.txt # لیست کتابخانه‌های مورد نیاز
└── PSDR/ # دیتاست اصلی


---


Epoch 5: Loss = 1.8424, Test Accuracy = 35.00%
Epoch 15: Loss = 1.0410, Test Accuracy = 60.56%
Epoch 35: Loss = 0.5698, Test Accuracy = 71.67%
Epoch 55: Loss = 0.3899, Test Accuracy = 79.44%

 دقت نهایی روی داده تست: 79.44%

