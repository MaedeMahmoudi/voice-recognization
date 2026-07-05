import librosa
import numpy as np
import os
import pickle
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "PSDR" / "dataset"
OUTPUT_PATH = BASE_DIR / "processed_data.pkl"

if not DATASET_PATH.exists():
    print("there is no dataset")
    exit()
print("dataset found")

def extract_mfcc_adaptive(file_path, n_mfcc=13):
    
    try:
        
        signal, sr = librosa.load(file_path, sr=22050)
        
        # حذف سکوت‌های ابتدا و انتها
        signal, _ = librosa.effects.trim(signal, top_db=20)
        
      
        mfcc = librosa.feature.mfcc(
            y=signal, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512,
            fmin=0,
            fmax=8000
        )
        
       
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        
        if mfcc_combined.shape[1] == 0:
            return None
            
        return mfcc_combined.T
        
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None

def main():
  
    all_features = []
    all_labels = []
    
    for digit_folder in sorted(os.listdir(DATASET_PATH)):
        digit_path = DATASET_PATH / digit_folder
        
        if not digit_path.is_dir():
            continue
        
        try:
            digit = int(digit_folder)
        except ValueError:
            continue
        
        wav_files = list(digit_path.glob("*.wav"))
        
        if len(wav_files) == 0:
            continue
        
        
        for wav_file in tqdm(wav_files, desc=f"digit {digit}"):
            mfcc_features = extract_mfcc_adaptive(str(wav_file))
            
            if mfcc_features is not None and len(mfcc_features) > 0:
                all_features.append(mfcc_features)
                all_labels.append(digit)
    
    if len(all_features) == 0:
        print("\n no data proceed")
        return
    
    print(f"\n all samples: {len(all_features)}")
    
    # ذخیره داده‌ها (بدون نرمالایز کردن)
    data = {
        'features': all_features,
        'labels': all_labels,
        'num_classes': 10
    }
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"saved {OUTPUT_PATH}")

if __name__ == "__main__":
    main()