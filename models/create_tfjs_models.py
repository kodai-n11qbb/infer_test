#!/usr/bin/env python3
"""
rc_system.py のロジックを TensorFlow.js モデルとして保存するスクリプト
辛口解説: rc_system.py は ML モデルを使っていないので、直接保存できるモデルはない。
このスクリプトは、rc_system.py のルールベースのロジックをトレーニングデータとして生成し、
TensorFlow モデルをトレーニングして TensorFlow.js 形式でエクスポートする。
これにより、ブラウザで推論可能になる。
"""

import numpy as np
import json
import os
import sys

# rc_system.py の設定をハードコード (本来は import すべきだが、独立させる)
# 距離推定設定
very_close_threshold = 10000
close_threshold = 5000
medium_threshold = 2000
far_threshold = 500

very_close_distance = 2.0
close_distance = 5.0
medium_distance = 10.0
far_distance = 15.0
very_far_distance = 20.0

# 分類設定
large_vehicle_threshold = 8000
vehicle_threshold = 3000
hazard_hue_threshold = 10

def generate_distance_training_data(num_samples=10000):
    """
    rc_system.py の距離推定ロジックに基づいてトレーニングデータを生成
    面積をランダムに生成し、ルールに基づいて距離を割り当てる
    """
    areas = np.random.uniform(100, 20000, num_samples)  # 面積の範囲
    distances = []

    for area in areas:
        if area > very_close_threshold:
            dist = very_close_distance
        elif area > close_threshold:
            dist = close_distance
        elif area > medium_threshold:
            dist = medium_distance
        elif area > far_threshold:
            dist = far_distance
        else:
            dist = very_far_distance
        distances.append(dist)

    # ノイズを追加して現実的に (ルールベースなのでノイズは小さい)
    distances = np.array(distances) + np.random.normal(0, 0.5, num_samples)

    return areas.reshape(-1, 1), distances.reshape(-1, 1)

def create_distance_model():
    """
    距離推定モデルを作成: 入力面積, 出力距離
    シンプルな NN でルールベースのステップ関数を近似
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # 回帰
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def generate_classification_training_data(num_samples=10000):
    """
    rc_system.py の分類ロジックに基づいてトレーニングデータを生成
    面積と HSV をランダム生成し、ラベルを割り当てる
    """
    areas = np.random.uniform(100, 20000, num_samples)
    hues = np.random.uniform(0, 180, num_samples)
    sats = np.random.uniform(0, 255, num_samples)
    vals = np.random.uniform(0, 255, num_samples)

    labels = []
    for area, hue in zip(areas, hues):
        if area > large_vehicle_threshold:
            label = 0  # large_vehicle
        elif area > vehicle_threshold:
            label = 1  # vehicle
        elif hue < hazard_hue_threshold or hue > (180 - hazard_hue_threshold):
            label = 2  # hazard
        else:
            label = 3  # object
        labels.append(label)

    X = np.column_stack([areas, hues, sats, vals])
    y = np.array(labels)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=4)

    return X, y_onehot

def create_classification_model():
    """
    物体分類モデルを作成: 入力面積+H+S+V, 出力4クラスの確率
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 多クラス分類
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def patch_tfjs_layers_model_json(model_json_path: str):
    """tfjs layers-model の model.json をブラウザ互換のキーに補正する。

    背景:
    - Keras v3 系から出た JSON は InputLayer の config に `batch_shape` が入ることがある
    - TFJS のデシリアライザは `batchInputShape` または `inputShape` を期待するためロードで落ちる
    """

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(model_json_path)

    with open(model_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # layers-model 以外なら何もしない
    if data.get('format') != 'layers-model':
        return

    model_topology = data.get('modelTopology') or {}
    model_config = (model_topology.get('model_config') or {}).get('config') or {}
    model_name = model_config.get('name')
    layers = model_config.get('layers') or []

    changed = False
    for layer in layers:
        if layer.get('class_name') != 'InputLayer':
            continue

        cfg = layer.get('config') or {}
        if 'batch_shape' in cfg and 'batchInputShape' not in cfg:
            cfg['batchInputShape'] = cfg.get('batch_shape')
            del cfg['batch_shape']
            changed = True

        if 'batchInputShape' in cfg and 'inputShape' in cfg:
            del cfg['inputShape']
            changed = True

        layer['config'] = cfg

    # weightsManifest 側の重み名も補正
    # 例: 'sequential/dense/kernel' -> 'dense/kernel'
    # TFJS 側はモデル名プレフィックス無しで変数を作ることがあり、ここがズレるとロードで落ちる。
    weights_manifest = data.get('weightsManifest') or []
    if model_name and weights_manifest:
        prefix = model_name + '/'
        for group in weights_manifest:
            weights = group.get('weights') or []
            for w in weights:
                name = w.get('name')
                if isinstance(name, str) and name.startswith(prefix):
                    w['name'] = name[len(prefix):]
                    changed = True

    if changed:
        with open(model_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def main():
    if '--patch-only' in sys.argv:
        patch_tfjs_layers_model_json('models/distance_model/model.json')
        patch_tfjs_layers_model_json('models/classification_model/model.json')
        print('patched model.json files')
        return

    import tensorflow as tf
    import tensorflowjs as tfjs

    print("rc_system.py のロジックを TensorFlow.js モデルに変換開始")

    # 距離推定モデル
    print("距離推定モデルのトレーニングデータ生成")
    X_dist, y_dist = generate_distance_training_data()

    print("距離推定モデル作成とトレーニング")
    dist_model = create_distance_model()
    dist_model.fit(X_dist, y_dist, epochs=100, batch_size=32, verbose=1)

    # 分類モデル
    print("分類モデルのトレーニングデータ生成")
    X_class, y_class = generate_classification_training_data()

    print("分類モデル作成とトレーニング")
    class_model = create_classification_model()
    class_model.fit(X_class, y_class, epochs=100, batch_size=32, verbose=1)

    # モデル保存ディレクトリ作成
    os.makedirs('models/distance_model', exist_ok=True)
    os.makedirs('models/classification_model', exist_ok=True)

    # SavedModel 形式で保存 (tensorflowjs_converter 用)
    print("距離推定モデルを SavedModel 形式で保存")
    dist_model.export('models/distance_model/saved_model')

    print("分類モデルを SavedModel 形式で保存")
    class_model.export('models/classification_model/saved_model')

    # TensorFlow.js 形式でエクスポート
    print("TensorFlow.js 形式に変換")
    tfjs.converters.save_keras_model(dist_model, 'models/distance_model')
    tfjs.converters.save_keras_model(class_model, 'models/classification_model')

    # ブラウザの TFJS でロードできる形に補正
    patch_tfjs_layers_model_json('models/distance_model/model.json')
    patch_tfjs_layers_model_json('models/classification_model/model.json')

    print("モデル変換完了: models/distance_model/model.json と models/classification_model/model.json が作成されました")
    print("realtime_inference.html でロード可能です")

if __name__ == "__main__":
    main()
