"""
TensorFlow.js 用モデルエクスポートスクリプト

現在の rc_system の設定に基づいて、距離推定と物体分類のモデルを生成し、TFJS 形式で出力する。

必要なライブラリ: tensorflow, tensorflowjs
インストール: pip install tensorflow tensorflowjs
"""

import tomllib
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def check_dependencies():
    """依存ライブラリをチェック"""
    try:
        import tensorflow as tf
        import tensorflowjs as tfjs
        print("依存ライブラリ OK")
        return tf, tfjs
    except ImportError as e:
        print(f"依存ライブラリが不足: {e}")
        print("pip install tensorflow tensorflowjs を実行してください")
        return None, None

def generate_distance_data(distance_estimation_config):
    """距離推定の訓練データを生成 (面積 -> 距離 のルールを近似)"""
    # 面積範囲: 300 - 50000
    areas = np.linspace(300, 50000, 1000)
    distances = []
    for area in areas:
        # 現在のルールを適用
        if area > distance_estimation_config['very_close_threshold']:
            dist = distance_estimation_config['very_close_distance']
        elif area > distance_estimation_config['close_threshold']:
            dist = distance_estimation_config['close_distance']
        elif area > distance_estimation_config['medium_threshold']:
            dist = distance_estimation_config['medium_distance']
        elif area > distance_estimation_config['far_threshold']:
            dist = distance_estimation_config['far_distance']
        else:
            dist = distance_estimation_config['very_far_distance']
        distances.append(dist)
    return areas.reshape(-1, 1), np.array(distances).reshape(-1, 1)

def create_distance_model(distance_estimation_config, tf):
    """距離推定モデルを作成 (線形回帰で近似)"""
    X, y = generate_distance_data(distance_estimation_config)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    return model

def generate_classification_data(object_classification_config):
    """物体分類の訓練データを生成 (面積, H -> タイプ)"""
    n_samples = 10000
    areas = np.random.uniform(300, 50000, n_samples)
    h_values = np.random.uniform(0, 180, n_samples)  # HSV H
    # S, V は固定
    s_values = np.full(n_samples, 128)
    v_values = np.full(n_samples, 128)
    labels = []
    for area, h in zip(areas, h_values):
        if area > object_classification_config['large_vehicle_threshold']:
            label = 0  # large_vehicle
        elif area > object_classification_config['vehicle_threshold']:
            label = 1  # vehicle
        elif h < object_classification_config['hazard_hue_threshold'] or h > (180 - object_classification_config['hazard_hue_threshold']):
            label = 2  # hazard
        else:
            label = 3  # object
        labels.append(label)
    X = np.column_stack([areas, h_values, s_values, v_values])
    y = np.array(labels)
    return X, y

def create_classification_model(object_classification_config, tf):
    """物体分類モデルを作成 (MLP 分類)"""
    X, y = generate_classification_data(object_classification_config)
    y = tf.one_hot(y, depth=4).numpy()
    # 手動で train/test 分割 (80% train)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
    return model

def main():
    tf, tfjs = check_dependencies()
    if not tf or not tfjs:
        return
    # infer.toml を読み込む
    with open('infer.toml', 'rb') as f:
        config = tomllib.load(f)
    # models ディレクトリを作成
    os.makedirs('models', exist_ok=True)
    # 距離推定モデル
    print("距離推定モデルを作成中...")
    dist_model = create_distance_model(config['distance_estimation'], tf)
    tfjs.converters.save_keras_model(dist_model, 'models/distance_model')
    print("距離推定モデル保存完了: models/distance_model")
    # 物体分類モデル
    print("物体分類モデルを作成中...")
    class_model = create_classification_model(config['object_classification'], tf)
    tfjs.converters.save_keras_model(class_model, 'models/classification_model')
    print("物体分類モデル保存完了: models/classification_model")

if __name__ == '__main__':
    main()
