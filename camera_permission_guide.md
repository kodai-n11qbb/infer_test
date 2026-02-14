# macOSカメラ権限設定ガイド

## 問題
OpenCVがカメラにアクセスできないエラーが発生しています。

## 解決方法

### 1. システム設定で権限を許可

1. **システム環境設定**を開く
2. **セキュリティとプライバシー**を選択
3. 左側メニューから**カメラ**を選択
4. リストからPythonまたはターミナルを見つけて**チェックを入れる**
5. すでにチェックがある場合は一度外して再度チェックする

### 2. ターミナルの再起動
```bash
# ターミナルを完全に終了して再起動
# または新しいターミナルウィンドウを開く
```

### 3. 他のアプリの確認
- Zoom、FaceTime、Skypeなどがカメラを使用していないか確認
- 使用中の場合は終了する

### 4. Pythonの再実行
```bash
cd /Users/abekoudai/Desktop/infer_test
source ./venv/bin/activate
python autonomous_driver.py
```

## それでも解決しない場合

### カメラIDの確認
```python
import cv2

# 利用可能なカメラデバイスを確認
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"カメラID {i} は利用可能です")
        cap.release()
    else:
        print(f"カメラID {i} は利用できません")
```

### 仮想カメラの使用
外部カメラがない場合は仮想カメラアプリを検討：
- OBS Virtual Camera
- ManyCam
- Camo

## 注意点
- macOSのセキュリティ機能により、初回実行時のみ権限ダイアログが表示されます
- 権限を拒否した場合は手動で設定が必要です
