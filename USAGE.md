# ラジコン式進行方向指示システム - 使用方法

## 概要
PCカメラから前方の障害物を検出し、安全距離に基づいて進行可否を指示するシステム。

## 特徴
- 外部から安全距離を設定可能
- リアルタイムの障害物検出と距離推定
- 直感的なUI（矢印表示）
- 設定ファイルとコマンドライン引数に対応

## 基本使用方法

### デフォルト設定で実行
```bash
python rc_system.py
```

### コマンドラインで距離を設定
```bash
# 緊急停止3m、注意8m、安全12m、検知範囲18mに設定
python rc_system.py --emergency 3.0 --caution 8.0 --safe 12.0 --range 18.0

# 屋内用（短距離）設定
python rc_system.py --emergency 1.5 --caution 3.0 --safe 5.0 --range 8.0

# 屋外用（長距離）設定
python rc_system.py --emergency 8.0 --caution 15.0 --safe 25.0 --range 40.0
```

### 設定ファイルを使用
```bash
# 設定をファイルに保存
python rc_system.py --emergency 3.0 --caution 8.0 --safe 12.0 --range 18.0 --save-config

# 保存した設定ファイルを使用
python rc_system.py --config my_config.json

# カスタム設定ファイル名
python rc_system.py --config indoor_settings.json
```

## 設定パラメータ

| パラメータ | 説明 | デフォルト | 推奨値 |
|-----------|------|----------|--------|
| `--emergency` | 緊急停止距離 | 5.0m | 屋内:1.5-3.0m, 屋外:5.0-10.0m |
| `--caution` | 注意距離 | 10.0m | 屋内:3.0-8.0m, 屋外:10.0-20.0m |
| `--safe` | 安全距離 | 15.0m | 屋内:5.0-12.0m, 屋外:15.0-30.0m |
| `--range` | 検知範囲 | 20.0m | 屋内:8.0-18.0m, 屋外:20.0-50.0m |

## UI表示内容

### 中央矢印
- **緑↑**: GO - 安全に前進可能
- **黄色↑**: CAREFUL - 注意しながら前進
- **橙色↓**: SLOW - 減速して注意
- **赤×**: STOP - 緊急停止

### 上部ステータスバー
- 進行可否ステータス（SAFE/CAUTION/DANGER）
- 判断理由の詳細表示

### 下部情報パネル
- 現在の安全距離設定
- 検出された障害物情報
- システム性能情報（FPS、実行時間）

### 障害物表示
- 距離に応じた色分け：
  - 赤: 緊急距離内
  - オレンジ: 注意距離内
  - 黄色: 安全距離内
  - 黄緑: 検知範囲内

## 使用シーン別推奨設定

### 室内・狭い空間
```bash
python rc_system.py --emergency 1.5 --caution 3.0 --safe 5.0 --range 8.0
```

### 一般室内
```bash
python rc_system.py --emergency 2.5 --caution 5.0 --safe 8.0 --range 12.0
```

### 屋外・広い空間
```bash
python rc_system.py --emergency 8.0 --caution 15.0 --safe 25.0 --range 40.0
```

### 高速移動対応
```bash
python rc_system.py --emergency 15.0 --caution 30.0 --safe 50.0 --range 80.0
```

## 設定ファイル形式

JSON形式で設定を保存：
```json
{
  "emergency_distance": 5.0,
  "caution_distance": 10.0,
  "safe_distance": 15.0,
  "detection_range": 20.0
}
```

## 操作方法
- **ESCキー**: システム終了
- **矢印表示**: 進行すべき方向を指示
- **距離情報**: リアルタイムで障害物までの距離を表示

## 注意事項
- カメラ権限を許可してください
- 十分な光量のある環境で使用してください
- 設定距離は使用環境に応じて調整してください
- 実際の運転制御には使用しないでください

## トラブルシューティング

### カメラが起動しない
1. macOS設定 > プライバシー > カメラで権限を許可
2. 他のカメラアプリを終了

### 障害物が検出されない
1. 光量を増やす
2. 検知範囲を広げる（--rangeを増やす）
3. 明るい色の物体を対象にする

### 感度が鈍い/鋭すぎる
1. 緊急距離を調整（--emergency）
2. 注意距離を調整（--caution）
3. 安全距離を調整（--safe）
