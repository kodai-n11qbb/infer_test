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
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_tfjs_model_structure(model, output_dir):
    """
    TensorFlowをインポートせずに、KerasモデルからTFJS Layers形式のmodel.jsonを生成する。
    TFJSのInputLayerエラーを回避するため、完全に互換性のある構造を手動で構築する。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 重みManifestの作成
    weights = model.get_weights()
    weights_binary = b""
    weight_entries = []
    
    # Sequentialモデルのレイヤー構成を取得
    # 手動で簡略化したLayers構成を構築（InputLayerエラー回避用）
    layers_config = []
    weight_idx = 0
    
    # 各レイヤーの構成を取得して変換
    for i, layer in enumerate(model.layers):
        l_config = layer.get_config()
        class_name = layer.__class__.__name__
        
        # 最初のレイヤーに入力形状を注入 (model.build()後の値を参照)
        if i == 0:
            # Sequentialモデルのbuild済み形状から取得
            l_config["batch_input_shape"] = [None] + list(model.input_shape[1:])
        
        # 【重要】dtypeがオブジェクト（DTypePolicy等）の場合、TFJSが解釈できず [object Object] エラーになるため、
        # 強制的に単純な文字列 "float32" に置換する
        if "dtype" in l_config:
            l_config["dtype"] = "float32"
        
        layers_config.append({
            "class_name": class_name,
            "config": l_config
        })
        
        # 重み情報の処理
        layer_weights = layer.get_weights()
        if layer_weights:
            # kernel
            kernel = layer_weights[0]
            kernel_name = f"{layer.name}/kernel"
            weight_entries.append({
                "name": kernel_name,
                "shape": list(kernel.shape),
                "dtype": "float32"
            })
            weights_binary += kernel.astype('float32').tobytes()
            
            # bias
            if len(layer_weights) > 1:
                bias = layer_weights[1]
                bias_name = f"{layer.name}/bias"
                weight_entries.append({
                    "name": bias_name,
                    "shape": list(bias.shape),
                    "dtype": "float32"
                })
                weights_binary += bias.astype('float32').tobytes()

    # TFJS形式のmodel.jsonの構築
    tfjs_model = {
        "format": "layers-model",
        "generatedBy": "custom-script",
        "convertedBy": None,
        "modelTopology": {
            "class_name": "Sequential",
            "config": {
                "name": "sequential",
                "layers": layers_config
            },
            "keras_version": "2.13.1", # 互換性の高いバージョンを明示
            "backend": "tensorflow"
        },
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": weight_entries
            }
        ]
    }
    
    # model.jsonの保存
    with open(os.path.join(output_dir, "model.json"), "w") as f:
        json.dump(tfjs_model, f, indent=2)
    
    # 重みバイナリの保存
    with open(os.path.join(output_dir, "group1-shard1of1.bin"), "wb") as f:
        f.write(weights_binary)

def check_dependencies():
    """依存ライブラリをチェック"""
    try:
        import tensorflow as tf
        print("TensorFlow OK")
        return tf
    except ImportError as e:
        print(f"TensorFlow がインストールされていません: {e}")
        return None

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
    tf = check_dependencies()
    if not tf:
        return
    # infer.toml を読み込む
    with open('infer.toml', 'rb') as f:
        config = tomllib.load(f)
    # models ディレクトリを作成
    os.makedirs('models', exist_ok=True)
    
    # 距離推定モデル
    print("距離推定モデルを作成中...")
    dist_model = create_distance_model(config['distance_estimation'], tf)
    create_tfjs_model_structure(dist_model, 'models/distance_model')
    print("距離推定モデル保存完了: models/distance_model")
    
    # 物体分類モデル
    print("物体分類モデルを作成中...")
    class_model = create_classification_model(config['object_classification'], tf)
    create_tfjs_model_structure(class_model, 'models/classification_model')
    print("物体分類モデル保存完了: models/classification_model")

if __name__ == '__main__':
    main()
