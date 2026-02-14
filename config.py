"""
自動運転システム設定ファイル
システムの動作パラメータを管理
"""

import cv2

# PCカメラ設定
CAMERA = {
    "camera_id": 0,           # カメラID（通常0が内蔵カメラ）
    "frame_width": 640,       # フレーム幅
    "frame_height": 480,      # フレーム高さ
    "fps": 30,               # 目標FPS
    "buffer_size": 1         # カメラバッファサイズ
}

# 車線検出設定
LANE_DETECTION = {
    "canny_low_threshold": 50,      # Cannyエッジ検出の下限閾値
    "canny_high_threshold": 150,    # Cannyエッジ検出の上限閾値
    "hough_threshold": 50,          # ハフ変換の投票閾値
    "min_line_length": 40,          # 検出する線分の最小長
    "max_line_gap": 20,             # 線分間の最大ギャップ
    "slope_threshold": 0.5,         # 車線傾きの閾値
    "roi_height_ratio": 0.6         # 関心領域の高さ比率
}

# 物体検出設定
OBJECT_DETECTION = {
    "model_path": "yolov8n.pt",           # YOLOモデルファイル
    "confidence_threshold": 0.5,          # 検出信頼度閾値
    "target_classes": [                   # 検出対象クラス
        "car", "truck", "bus", "motorcycle", "bicycle", "person"
    ],
    "reference_sizes": {                   # 距離推定用基準サイズ
        "car": 100,
        "truck": 150,
        "bus": 180,
        "motorcycle": 60,
        "bicycle": 50,
        "person": 40
    },
    "max_detection_distance": 100.0,       # 最大検出距離（メートル）
    "min_detection_distance": 1.0          # 最小検出距離（メートル）
}

# 意思決定設定
DECISION_MAKING = {
    "emergency_brake_distance": 5.0,      # 緊急ブレーキ距離（メートル）
    "safe_following_distance": 15.0,       # 安全車間距離（メートル）
    "lane_deviation_threshold": 50,        # 車線逸脱閾値（ピクセル）
    "front_detection_ratio": 0.3,          # 前方検出領域比率
    "instruction_history_size": 10,        # 指示履歴サイズ
    "urgency_levels": {                    # 緊急度レベル定義
        "emergency": 10,    # 緊急事態
        "high": 8,          # 高い緊急度
        "medium": 5,        # 中程度の緊急度
        "low": 2            # 低い緊急度
    }
}

# システム全体設定
SYSTEM = {
    "target_fps": 30,                       # 目標FPS
    "max_frame_time": 1.0/30,              # 最大フレーム時間
    "window_name": "Autonomous Driving System",  # 表示ウィンドウ名
    "exit_key": 27,                        # 終了キー（ESC）
    "enable_logging": True,                 # ログ出力有効化
    "log_level": "INFO"                    # ログレベル
}

# 可視化設定
VISUALIZATION = {
    "lane_color": (0, 255, 0),             # 車線の色（緑）
    "lane_thickness": 3,                   # 車線の太さ
    "object_color": (255, 0, 0),           # 物体の色（青）
    "object_thickness": 2,                 # 物体ボックスの太さ
    "text_font": cv2.FONT_HERSHEY_SIMPLEX, # テキストフォント
    "text_scale": 0.6,                     # テキストスケール
    "text_thickness": 2,                   # テキスト太さ
    "emergency_color": (0, 0, 255),       # 緊急時の色（赤）
    "warning_color": (0, 165, 255),        # 警告時の色（オレンジ）
    "normal_color": (0, 255, 0)            # 正常時の色（緑）
}
