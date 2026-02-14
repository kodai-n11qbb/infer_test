#!/usr/bin/env python3
"""
ラジコン式自動運転システム
前方方向への進行可否を指示するシンプルUI
外部から安全距離を設定可能
"""

import cv2
import numpy as np
import time
import argparse
import tomllib
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class SafetyConfig:
    """安全設定データクラス"""
    emergency_distance: float = 5.0    # 緊急ブレーキ距離（m）
    caution_distance: float = 10.0    # 注意距離（m）
    safe_distance: float = 15.0       # 安全距離（m）
    detection_range: float = 20.0     # 検知範囲（m）
    
    @classmethod
    def from_file(cls, filename: str = "safety_config.toml"):
        """設定ファイルから読み込み"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = tomllib.load(f)

                return cls(
                    emergency_distance=float(data.get('emergency_distance', 5.0)),
                    caution_distance=float(data.get('caution_distance', 10.0)),
                    safe_distance=float(data.get('safe_distance', 15.0)),
                    detection_range=float(data.get('detection_range', 20.0)),
                )
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
                return cls()
        return cls()
    
    def to_file(self, filename: str = "safety_config.toml"):
        """設定ファイルに保存"""
        try:
            # tomllib は読み込み専用なので、最小限のTOMLを自前で書き出す
            content = (
                f"emergency_distance = {self.emergency_distance}\n"
                f"caution_distance = {self.caution_distance}\n"
                f"safe_distance = {self.safe_distance}\n"
                f"detection_range = {self.detection_range}\n"
            )
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"設定を {filename} に保存しました")
        except Exception as e:
            print(f"設定ファイル保存エラー: {e}")

@dataclass
class InferConfig:
    """設定ファイルから読み込む全設定データクラス"""
    safety_distances: SafetyConfig
    detection_settings: Dict[str, Any]
    ui_settings: Dict[str, Any]
    camera_settings: Dict[str, Any]
    color_ranges: Dict[str, Any]
    object_classification: Dict[str, Any]
    distance_estimation: Dict[str, Any]
    
    @classmethod
    def from_infer_file(cls, filename: str = "infer.toml"):
        """
        設定ファイルから全設定を読み込み
        外部設定ファイルでシステム全体を制御
        """
        if not os.path.exists(filename):
            print(f"警告: {filename} が見つかりません。デフォルト設定を使用します")
            return cls._get_default_config()
        
        try:
            with open(filename, 'rb') as f:
                data = tomllib.load(f)
            
            # 安全距離設定をSafetyConfigオブジェクトに変換
            safety_data = data.get('safety_distances', {})
            safety_config = SafetyConfig(
                emergency_distance=safety_data.get('emergency_distance', 5.0),
                caution_distance=safety_data.get('caution_distance', 10.0),
                safe_distance=safety_data.get('safe_distance', 15.0),
                detection_range=safety_data.get('detection_range', 20.0)
            )
            
            return cls(
                safety_distances=safety_config,
                detection_settings=data.get('detection_settings', {}),
                ui_settings=data.get('ui_settings', {}),
                camera_settings=data.get('camera_settings', {}),
                color_ranges=data.get('color_ranges', {}),
                object_classification=data.get('object_classification', {}),
                distance_estimation=data.get('distance_estimation', {})
            )
            
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            print("デフォルト設定を使用します")
            return cls._get_default_config()
    
    @classmethod
    def _get_default_config(cls):
        """デフォルト設定を返す"""
        safety_config = SafetyConfig()
        return cls(
            safety_distances=safety_config,
            detection_settings={},
            ui_settings={},
            camera_settings={},
            color_ranges={},
            object_classification={},
            distance_estimation={}
        )
    
    def validate(self):
        """設定の妥当性を検証"""
        safety = self.safety_distances
        
        # 距離設定の妥当性チェック
        if not (0 < safety.emergency_distance < safety.caution_distance < 
                safety.safe_distance < safety.detection_range):
            print("エラー: 距離設定が不正です")
            print("緊急 < 注意 < 安全 < 検知範囲 の順に設定してください")
            return False
        
        return True

class SimpleDetector:
    """シンプルな障害物検出器 - infer.toml設定対応"""
    
    def __init__(self, config: InferConfig):
        """
        検出器を初期化 - 外部設定ファイルからパラメータを読み込み
        """
        self.config = config
        self.frame_count = 0
        
        # 検出設定を取得
        self.confidence_threshold = config.detection_settings.get('confidence_threshold', 0.5)
        self.min_object_area = config.detection_settings.get('min_object_area', 300)
        self.max_object_area = config.detection_settings.get('max_object_area', 50000)
        self.aspect_ratio_min = config.detection_settings.get('aspect_ratio_min', 0.3)
        self.aspect_ratio_max = config.detection_settings.get('aspect_ratio_max', 3.0)
        
        # 距離推定設定を取得
        dist_config = config.distance_estimation
        self.very_close_threshold = dist_config.get('very_close_threshold', 10000)
        self.close_threshold = dist_config.get('close_threshold', 5000)
        self.medium_threshold = dist_config.get('medium_threshold', 2000)
        self.far_threshold = dist_config.get('far_threshold', 500)
        
        self.very_close_distance = dist_config.get('very_close_distance', 2.0)
        self.close_distance = dist_config.get('close_distance', 5.0)
        self.medium_distance = dist_config.get('medium_distance', 10.0)
        self.far_distance = dist_config.get('far_distance', 15.0)
        self.very_far_distance = dist_config.get('very_far_distance', 20.0)
        
        # 物体分類設定を取得
        class_config = config.object_classification
        self.large_vehicle_threshold = class_config.get('large_vehicle_threshold', 8000)
        self.vehicle_threshold = class_config.get('vehicle_threshold', 3000)
        self.hazard_hue_threshold = class_config.get('hazard_hue_threshold', 10)
        
        print(f"検出器設定:")
        print(f"- 信頼度閾値: {self.confidence_threshold}")
        print(f"- 物体面積範囲: {self.min_object_area}-{self.max_object_area}")
        print(f"- アスペクト比範囲: {self.aspect_ratio_min}-{self.aspect_ratio_max}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        障害物を検出 - infer.tomlの設定を使用
        戻り値: 検出された障害物リスト
        """
        self.frame_count += 1
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # infer.tomlから色範囲を取得
        color_ranges = self.config.color_ranges
        if not color_ranges:
            # デフォルト色範囲
            color_ranges = {
                'red': [[0, 50, 50], [10, 255, 255]],
                'red2': [[170, 50, 50], [180, 255, 255]],
                'blue': [[100, 50, 50], [130, 255, 255]],
                'yellow': [[20, 50, 50], [30, 255, 255]]
            }
        
        # 全色マスクを統合
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_name, ranges in color_ranges.items():
            if isinstance(ranges[0], list) and len(ranges) == 2:
                lower = np.array(ranges[0])
                upper = np.array(ranges[1])
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # infer.tomlの設定でフィルタリング
            if area < self.min_object_area or area > self.max_object_area:
                continue
            
            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(contour)
            
            # アスペクト比でフィルタリング
            aspect_ratio = w / h
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                continue
            
            # 距離推定（infer.tomlの設定を使用）
            distance = self._estimate_distance(w, h)
            
            # 物体タイプ判定（infer.tomlの設定を使用）
            obj_type = self._classify_object(hsv[y:y+h, x:x+w], w, h)
            
            # 信頼度計算
            confidence = min(0.9, area / 5000)
            if confidence < self.confidence_threshold:
                continue
            
            objects.append({
                'type': obj_type,
                'bbox': [x, y, x + w, y + h],
                'distance': distance,
                'confidence': confidence,
                'area': area
            })
        
        return objects
    
    def _estimate_distance(self, width: int, height: int) -> float:
        """
        物体サイズから距離を推定 - infer.tomlの設定を使用
        """
        area = width * height
        
        # infer.tomlの閾値設定を使用
        if area > self.very_close_threshold:
            return self.very_close_distance
        elif area > self.close_threshold:
            return self.close_distance
        elif area > self.medium_threshold:
            return self.medium_distance
        elif area > self.far_threshold:
            return self.far_distance
        else:
            return self.very_far_distance
    
    def _classify_object(self, roi_hsv: np.ndarray, width: int, height: int) -> str:
        """
        物体タイプを分類 - infer.tomlの設定を使用
        """
        # 平均色を取得
        mean_color = np.mean(roi_hsv, axis=(0, 1))
        
        # サイズと色で分類（infer.tomlの閾値を使用）
        area = width * height
        
        if area > self.large_vehicle_threshold:
            return "large_vehicle"
        elif area > self.vehicle_threshold:
            return "vehicle"
        elif mean_color[0] < self.hazard_hue_threshold or mean_color[0] > (180 - self.hazard_hue_threshold):
            return "hazard"
        else:
            return "object"

class DirectionDecider:
    """進行方向決定器 - 設定された安全距離に基づいて判断"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.decision_count = 0
    
    def decide_direction(self, objects: List[Dict]) -> Dict[str, Any]:
        """
        安全距離設定に基づいて進行方向を決定
        戻り値: 進行指示情報
        """
        self.decision_count += 1
        
        # 前方領域の障害物を抽出
        frame_center_x = 320  # 640pxの中心
        front_objects = []
        
        for obj in objects:
            obj_center_x = (obj['bbox'][0] + obj['bbox'][2]) // 2
            # 前方±100pxを前方領域とする
            if abs(obj_center_x - frame_center_x) < 100:
                front_objects.append(obj)
        
        # 最も近い前方障害物
        closest_front = None
        if front_objects:
            closest_front = min(front_objects, key=lambda x: x['distance'])
        
        # 安全距離に基づく判断
        if closest_front:
            distance = closest_front['distance']
            
            if distance <= self.config.emergency_distance:
                # 緊急停止
                return {
                    'action': 'STOP',
                    'urgency': 10,
                    'reason': f"緊急：{closest_front['type']}が{distance:.1f}m（設定: {self.config.emergency_distance}m）",
                    'confidence': 0.95,
                    'distance': distance
                }
            elif distance <= self.config.caution_distance:
                # 減速・注意
                return {
                    'action': 'SLOW',
                    'urgency': 6,
                    'reason': f"注意：{closest_front['type']}が{distance:.1f}m（設定: {self.config.caution_distance}m）",
                    'confidence': 0.8,
                    'distance': distance
                }
            elif distance <= self.config.safe_distance:
                # 注意しながら前進
                return {
                    'action': 'CAREFUL',
                    'urgency': 3,
                    'reason': f"安全圏内：{closest_front['type']}が{distance:.1f}m（設定: {self.config.safe_distance}m）",
                    'confidence': 0.7,
                    'distance': distance
                }
        
        # 安全な前進
        return {
            'action': 'GO',
            'urgency': 1,
            'reason': f"前方安全：障害物なし（検知範囲: {self.config.detection_range}m）",
            'confidence': 0.9,
            'distance': None
        }

class FinalInstructionDecider:
    """最終指示（前進/後進/右/左/停止）を確定する"""

    def __init__(self, safety_config: SafetyConfig, frame_width: int = 640):
        self.config = safety_config
        self.frame_width = frame_width

    def decide(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """停止/後進/右/左/前進 のいずれかを返す"""

        left_bound = self.frame_width // 3
        right_bound = (self.frame_width * 2) // 3
        front_center = self.frame_width // 2

        left_objects: List[Dict[str, Any]] = []
        front_objects: List[Dict[str, Any]] = []
        right_objects: List[Dict[str, Any]] = []

        for obj in objects:
            x1, _, x2, _ = obj['bbox']
            obj_center_x = (x1 + x2) // 2

            if abs(obj_center_x - front_center) < 100:
                front_objects.append(obj)
            elif obj_center_x < left_bound:
                left_objects.append(obj)
            elif obj_center_x > right_bound:
                right_objects.append(obj)

        closest_front = min(front_objects, key=lambda x: x['distance']) if front_objects else None
        closest_left = min(left_objects, key=lambda x: x['distance']) if left_objects else None
        closest_right = min(right_objects, key=lambda x: x['distance']) if right_objects else None

        if closest_front and closest_front['distance'] <= self.config.emergency_distance:
            return {
                'final_action': '停止',
                'urgency': 10,
                'reason': f"緊急停止：前方{closest_front['type']}が{closest_front['distance']:.1f}m（設定: {self.config.emergency_distance}m）",
                'confidence': 0.95,
            }

        if closest_front and closest_front['distance'] <= self.config.caution_distance:
            left_clearance = closest_left['distance'] if closest_left else float('inf')
            right_clearance = closest_right['distance'] if closest_right else float('inf')

            if left_clearance <= self.config.caution_distance and right_clearance <= self.config.caution_distance:
                return {
                    'final_action': '後進',
                    'urgency': 8,
                    'reason': f"回避不能：前方{closest_front['distance']:.1f}m & 左右も近接（設定: {self.config.caution_distance}m）",
                    'confidence': 0.85,
                }

            if right_clearance > left_clearance:
                return {
                    'final_action': '右',
                    'urgency': 7,
                    'reason': f"回避：右がより空いている（右={right_clearance:.1f}m, 左={left_clearance:.1f}m）",
                    'confidence': 0.8,
                }

            return {
                'final_action': '左',
                'urgency': 7,
                'reason': f"回避：左がより空いている（左={left_clearance:.1f}m, 右={right_clearance:.1f}m）",
                'confidence': 0.8,
            }

        if closest_front and closest_front['distance'] <= self.config.safe_distance:
            return {
                'final_action': '前進',
                'urgency': 3,
                'reason': f"慎重前進：前方{closest_front['type']}が{closest_front['distance']:.1f}m（設定: {self.config.safe_distance}m）",
                'confidence': 0.7,
            }

        return {
            'final_action': '前進',
            'urgency': 1,
            'reason': f"前方安全：障害物なし（検知範囲: {self.config.detection_range}m）",
            'confidence': 0.9,
        }

class RCController:
    """ラジコン式コントローラー - infer.toml完全対応"""
    
    def __init__(self, config: InferConfig):
        """
        コントローラーを初期化 - 外部設定ファイルで全システムを制御
        """
        self.config = config
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # カメラ設定を取得
        camera_config = config.camera_settings
        self.camera_id = camera_config.get('camera_id', 0)
        self.cap = None
        
        # UI設定を取得
        ui_config = config.ui_settings
        self.show_distance_labels = ui_config.get('show_distance_labels', True)
        self.show_confidence = ui_config.get('show_confidence', True)
        self.show_fps = ui_config.get('show_fps', True)
        self.arrow_size = ui_config.get('arrow_size', 60)
        self.blink_frequency = ui_config.get('blink_frequency', 4)
        
        # 検出器と意思決定器を設定で初期化
        self.detector = SimpleDetector(config)
        self.decider = DirectionDecider(config.safety_distances)
        self.final_decider = FinalInstructionDecider(config.safety_distances)
        
        print(f"infer.toml設定を読み込み:")
        print(f"- 緊急停止: {config.safety_distances.emergency_distance}m")
        print(f"- 注意距離: {config.safety_distances.caution_distance}m") 
        print(f"- 安全距離: {config.safety_distances.safe_distance}m")
        print(f"- 検知範囲: {config.safety_distances.detection_range}m")
        print(f"- カメラID: {self.camera_id}")
        print(f"- UI設定: ラベル表示={self.show_distance_labels}, 信頼度表示={self.show_confidence}")
    
    def start(self):
        """システム起動"""
        print("=" * 60)
        print("ラジコン式進行方向指示システム - infer.toml対応")
        print("=" * 60)
        print("特徴:")
        print("- infer.tomlで全設定を外部制御")
        print("- 前方方向への進行可否をリアルタイム表示")
        print("- シンプルなUIで直感的な操作")
        print("- ESCキーで終了")
        print("-" * 60)
        
        # カメラ初期化
        if not self._init_camera():
            return
        
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nシステムを停止します...")
        finally:
            self._cleanup()
    
    def _init_camera(self) -> bool:
        """
        カメラ初期化 - infer.tomlの設定を使用
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"カメラID {self.camera_id} を開けません")
            return False
        
        # infer.tomlのカメラ設定を適用
        camera_config = self.config.camera_settings
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        fps = camera_config.get('fps', 30)
        buffer_size = camera_config.get('buffer_size', 1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.final_decider.frame_width = actual_width
        
        print(f"カメラ初期化完了: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    
    def _main_loop(self):
        """メインループ"""
        while self.running:
            loop_start = time.time()
            
            # フレーム取得
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            
            self.frame_count += 1
            
            # 障害物検出
            objects = self.detector.detect(frame)
            
            # 進行方向決定
            decision = self.decider.decide_direction(objects)

            # 最終指示（5種）
            final_decision = self.final_decider.decide(objects)
            
            # FPS更新
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # UI描画
            self._draw_ui(frame, decision, final_decision, objects)
            
            # フレームレート制御
            process_time = time.time() - loop_start
            if process_time < 1/30:
                time.sleep(1/30 - process_time)
    
    def _draw_ui(self, frame: np.ndarray, decision: Dict[str, Any], final_decision: Dict[str, Any], objects: List[Dict]):
        """シンプルなUIを描画 - 最終指示（5種）"""
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # 背景を少し暗くしてUIを見やすく
        vis_frame = cv2.convertScaleAbs(vis_frame, alpha=0.8, beta=0)
        
        # === 中央の進行方向インジケーター ===
        center_x = width // 2
        center_y = height // 2

        final_action = final_decision['final_action']
        if final_action == "停止":
            arrow_color = (0, 0, 255)
            arrow_size = 80
            cv2.line(vis_frame, (center_x - arrow_size, center_y - arrow_size),
                     (center_x + arrow_size, center_y + arrow_size), arrow_color, 8)
            cv2.line(vis_frame, (center_x + arrow_size, center_y - arrow_size),
                     (center_x - arrow_size, center_y + arrow_size), arrow_color, 8)
        elif final_action == "後進":
            arrow_color = (0, 165, 255)
            self._draw_down_arrow(vis_frame, center_x, center_y, arrow_color)
        elif final_action == "左":
            arrow_color = (255, 255, 0)
            self._draw_left_arrow(vis_frame, center_x, center_y, arrow_color)
        elif final_action == "右":
            arrow_color = (255, 255, 0)
            self._draw_right_arrow(vis_frame, center_x, center_y, arrow_color)
        else:
            arrow_color = (0, 255, 0)
            self._draw_up_arrow(vis_frame, center_x, center_y, arrow_color)
        
        # === 方向テキスト表示 ===
        text_bg_color = (0, 0, 0)
        cv2.rectangle(vis_frame, (center_x - 60, center_y + 100), 
                      (center_x + 60, center_y + 140), text_bg_color, -1)
        direction_text = final_action
        cv2.putText(vis_frame, direction_text, (center_x - 40, center_y + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, arrow_color, 3)
        
        # === 上部ステータスバー ===
        # ステータス背景
        cv2.rectangle(vis_frame, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # 進行可否ステータス
        if final_decision['urgency'] >= 8:
            status_text = "DANGER"
            status_color = (0, 0, 255)
        elif final_decision['urgency'] >= 5:
            status_text = "CAUTION"
            status_color = (0, 165, 255)
        else:
            status_text = "SAFE"
            status_color = (0, 255, 0)

        cv2.putText(vis_frame, status_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 理由表示
        reason_text = decision['reason'][:40] + "..." if len(decision['reason']) > 40 else decision['reason']
        cv2.putText(vis_frame, reason_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # === 下部情報パネル ===
        # 情報背景
        cv2.rectangle(vis_frame, (0, height - 100), (width, height), (0, 0, 0), -1)
        
        # 安全距離設定情報
        config_text = f"Safe: {self.config.safety_distances.safe_distance}m | Caution: {self.config.safety_distances.caution_distance}m | Emergency: {self.config.safety_distances.emergency_distance}m"
        cv2.putText(vis_frame, config_text, (10, height - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 障害物情報
        danger_objects = [obj for obj in objects if obj['distance'] < self.config.safety_distances.detection_range]
        if danger_objects:
            closest = min(danger_objects, key=lambda x: x['distance'])
            obstacle_text = f"OBSTACLE: {closest['type']} ({closest['distance']:.1f}m)"
            obstacle_color = (0, 0, 255) if closest['distance'] < self.config.safety_distances.emergency_distance else (0, 165, 255)
        else:
            obstacle_text = f"OBSTACLE: None (Range: {self.config.safety_distances.detection_range}m)"
            obstacle_color = (0, 255, 0)
        
        cv2.putText(vis_frame, obstacle_text, (10, height - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, obstacle_color, 2)
        
        # システム情報
        info_text = f"FPS: {self.fps:.1f} | Time: {int(time.time() - self.start_time)}s | Confidence: {final_decision['confidence']:.2f}"
        cv2.putText(vis_frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # === 障害物の簡易表示 ===
        for obj in objects:
            if obj['distance'] < self.config.safety_distances.detection_range:  # 設定された検知範囲内のみ
                x1, y1, x2, y2 = obj['bbox']
                
                # 距離に応じた色（設定値に基づく）
                if obj['distance'] < self.config.safety_distances.emergency_distance:
                    box_color = (0, 0, 255)  # 赤：危険
                elif obj['distance'] < self.config.safety_distances.caution_distance:
                    box_color = (0, 165, 255)  # オレンジ：注意
                elif obj['distance'] < self.config.safety_distances.safe_distance:
                    box_color = (0, 255, 255)  # 黄色：安全圏内
                else:
                    box_color = (255, 255, 0)  # 黄緑：検知範囲内
                
                # 簡易ボックス
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # 距離ラベル
                dist_text = f"{obj['distance']:.1f}m"
                cv2.putText(vis_frame, dist_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        
        # === 緊急時の警告表示 ===
        if final_decision['urgency'] >= 8:
            # 点滅警告
            if int(time.time() * 4) % 2 == 0:
                # 画面全体に赤枠
                cv2.rectangle(vis_frame, (0, 0), (width, height), (0, 0, 255), 15)
                # 大きな警告テキスト
                cv2.putText(vis_frame, "!! DANGER !!", (center_x - 120, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        
        # === 制御ヒント ===
        hint_text = "ESC: Exit | Arrow shows final instruction"
        cv2.putText(vis_frame, hint_text, (10, height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # 表示
        cv2.imshow('RC Controller - Final Instruction (5 actions)', vis_frame)
        
        # ESCキーで終了
        if cv2.waitKey(1) & 0xFF == 27:
            self.running = False
    
    
    def _draw_up_arrow(self, frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
        """上向き矢印を描画"""
        size = 60
        # 矢印本体
        points = [
            (x, y - size),      # 先端
            (x - size//2, y),   # 左下
            (x - size//4, y),   # 左上
            (x - size//4, y + size//2),  # 左下
            (x + size//4, y + size//2),  # 右下
            (x + size//4, y),   # 右上
            (x + size//2, y)    # 右下
        ]
        cv2.fillPoly(frame, [np.array(points)], color)
    
    def _draw_down_arrow(self, frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
        """下向き矢印を描画"""
        size = 60
        # 矢印本体
        points = [
            (x, y + size),      # 先端
            (x - size//2, y),   # 左上
            (x - size//4, y),   # 左下
            (x - size//4, y - size//2),  # 左上
            (x + size//4, y - size//2),  # 右上
            (x + size//4, y),   # 右下
            (x + size//2, y)    # 右上
        ]
        cv2.fillPoly(frame, [np.array(points)], color)
    
    def _draw_left_arrow(self, frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
        """左向き矢印を描画"""
        size = 60
        # 矢印本体
        points = [
            (x - size, y),      # 先端
            (x, y - size//2),   # 左上
            (x, y - size//4),   # 右上
            (x + size//2, y - size//4),  # 右上
            (x + size//2, y + size//4),  # 右下
            (x, y + size//4),   # 右下
            (x, y + size//2)    # 左下
        ]
        cv2.fillPoly(frame, [np.array(points)], color)
    
    def _draw_right_arrow(self, frame: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
        """右向き矢印を描画"""
        size = 60
        # 矢印本体
        points = [
            (x + size, y),      # 先端
            (x, y - size//2),   # 左上
            (x, y - size//4),   # 左下
            (x - size//2, y - size//4),  # 左下
            (x - size//2, y + size//4),  # 左上
            (x, y + size//4),   # 左上
            (x, y + size//2)    # 右下
        ]
        cv2.fillPoly(frame, [np.array(points)], color)
    
    def _cleanup(self):
        """クリーンアップ"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("システムを停止しました")

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='ラジコン式進行方向指示システム - infer.toml対応')
    
    parser.add_argument('--config', type=str, default='infer.toml',
                       help='設定ファイルパス - デフォルト: infer.toml')
    parser.add_argument('--emergency', type=float,
                       help='緊急停止距離（メートル）- 設定を上書き')
    parser.add_argument('--caution', type=float,
                       help='注意距離（メートル）- 設定を上書き')
    parser.add_argument('--safe', type=float,
                       help='安全距離（メートル）- 設定を上書き')
    parser.add_argument('--range', type=float,
                       help='検知範囲（メートル）- 設定を上書き')
    
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_arguments()
    
    # 設定ファイルから設定を読み込み
    print(f"設定ファイル: {args.config}")
    config = InferConfig.from_infer_file(args.config)
    
    # 設定の妥当性を検証
    if not config.validate():
        print("設定検証に失敗しました")
        return
    
    # コマンドライン引数で安全距離を上書き（指定された場合のみ）
    if args.emergency is not None:
        config.safety_distances.emergency_distance = args.emergency
        print(f"緊急停止距離を上書き: {args.emergency}m")
    
    if args.caution is not None:
        config.safety_distances.caution_distance = args.caution
        print(f"注意距離を上書き: {args.caution}m")
    
    if args.safe is not None:
        config.safety_distances.safe_distance = args.safe
        print(f"安全距離を上書き: {args.safe}m")
    
    if args.range is not None:
        config.safety_distances.detection_range = args.range
        print(f"検知範囲を上書き: {args.range}m")
    
    # 上書き後の妥当性を再検証
    if not config.validate():
        print("上書き後の設定が不正です")
        return
    
    # 設定内容を表示
    print("\n=== 現在の設定 ===")
    print(f"安全距離: 緊急={config.safety_distances.emergency_distance}m, "
          f"注意={config.safety_distances.caution_distance}m, "
          f"安全={config.safety_distances.safe_distance}m, "
          f"範囲={config.safety_distances.detection_range}m")
    
    if config.detection_settings:
        print(f"検出設定: 信頼度閾値={config.detection_settings.get('confidence_threshold', 'N/A')}")
    
    if config.ui_settings:
        print(f"UI設定: 矢印サイズ={config.ui_settings.get('arrow_size', 'N/A')}")
    
    print("==================\n")
    
    # システム起動
    controller = RCController(config)
    controller.start()

if __name__ == "__main__":
    main()
