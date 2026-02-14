
// TensorFlow.js 用モデル (ルールベース移植)
// infer.toml から生成された設定に基づく距離推定と物体分類関数
// TFJS ではモデルロードが必要だが、ここではルール移植。

const config = {
  "safety_distances": {
    "emergency_distance": 1.0,
    "caution_distance": 5.0,
    "safe_distance": 15.0,
    "detection_range": 20.0
  },
  "detection_settings": {
    "confidence_threshold": 0.5,
    "min_object_area": 300,
    "max_object_area": 50000,
    "aspect_ratio_min": 0.3,
    "aspect_ratio_max": 3.0
  },
  "ui_settings": {
    "show_distance_labels": true,
    "show_confidence": true,
    "show_fps": true,
    "arrow_size": 60,
    "blink_frequency": 4
  },
  "camera_settings": {
    "camera_id": 0,
    "width": 640,
    "height": 480,
    "fps": 30,
    "buffer_size": 1
  },
  "color_ranges": {
    "red": [
      [
        0,
        50,
        50
      ],
      [
        10,
        255,
        255
      ]
    ],
    "red2": [
      [
        170,
        50,
        50
      ],
      [
        180,
        255,
        255
      ]
    ],
    "blue": [
      [
        100,
        50,
        50
      ],
      [
        130,
        255,
        255
      ]
    ],
    "yellow": [
      [
        20,
        50,
        50
      ],
      [
        30,
        255,
        255
      ]
    ]
  },
  "object_classification": {
    "large_vehicle_threshold": 8000,
    "vehicle_threshold": 3000,
    "hazard_hue_threshold": 10
  },
  "distance_estimation": {
    "very_close_threshold": 10000,
    "close_threshold": 5000,
    "medium_threshold": 2000,
    "far_threshold": 500,
    "very_close_distance": 2.0,
    "close_distance": 5.0,
    "medium_distance": 10.0,
    "far_distance": 15.0,
    "very_far_distance": 20.0
  }
};  // 設定を埋め込み

// 距離推定関数 (面積 -> 距離)
// 現在の _estimate_distance のルールを移植
function estimateDistance(area) {
    const thresholds = config.distance_estimation;
    if (area > thresholds.very_close_threshold) {
        return thresholds.very_close_distance;  // 非常に近い
    } else if (area > thresholds.close_threshold) {
        return thresholds.close_distance;  // 近い
    } else if (area > thresholds.medium_threshold) {
        return thresholds.medium_distance;  // 中間
    } else if (area > thresholds.far_threshold) {
        return thresholds.far_distance;  // 遠い
    } else {
        return thresholds.very_far_distance;  // 非常に遠い
    }
}

// 物体分類関数 (面積, H -> タイプ)
// 現在の _classify_object のルールを移植 (S, V は無視)
function classifyObject(area, hValue) {
    const classification = config.object_classification;
    if (area > classification.large_vehicle_threshold) {
        return 'large_vehicle';  // 大型車両
    } else if (area > classification.vehicle_threshold) {
        return 'vehicle';  // 車両
    } else if (hValue < classification.hazard_hue_threshold || hValue > (180 - classification.hazard_hue_threshold)) {
        return 'hazard';  // 危険物 (色相ベース)
    } else {
        return 'object';  // 一般物体
    }
}

// 使用例: 推論実行
const testArea = 5000;  // テスト面積
const testH = 100;      // テスト色相
console.log('距離:', estimateDistance(testArea), 'm');
console.log('タイプ:', classifyObject(testArea, testH));

// TFJS で使う場合: この関数を呼び出せばルールベース推論可能。
// ML モデルではないが、設定移植で対応。
