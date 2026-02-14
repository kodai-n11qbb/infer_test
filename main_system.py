#!/usr/bin/env python3
"""
メインシステム - モジュール化された自動運転システム
モデル推論部分を簡単に改造できる構造
"""

import cv2
import numpy as np
import time
from typing import Dict, Any
from models import LaneDetectionModel, ObjectDetectionModel, DecisionMakingModel
from models import LaneResult, DetectionResult, DrivingCommand

class AutonomousDrivingSystem:
    """自動運転システムのメインクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # カメラ設定
        self.camera_id = self.config.get('camera_id', 0)
        self.cap = None
        
        # モデル初期化 - ここでモデルを切り替え可能
        self._initialize_models()
    
    def _initialize_models(self):
        """モデルを初期化 - 設定ファイルでモデルを切り替え"""
        # 車線検出モデル
        lane_config = self.config.get('lane_detection', {})
        self.lane_model = LaneDetectionModel(lane_config)
        
        # 物体検出モデル
        object_config = self.config.get('object_detection', {'model_type': 'contour'})
        self.object_model = ObjectDetectionModel(object_config)
        
        # 意思決定モデル
        decision_config = self.config.get('decision_making', {})
        self.decision_model = DecisionMakingModel(decision_config)
        
        print("モデル初期化完了:")
        print(f"- 車線検出: {type(self.lane_model).__name__}")
        print(f"- 物体検出: {type(self.object_model).__name__} ({self.object_model.model_type})")
        print(f"- 意思決定: {type(self.decision_model).__name__}")
    
    def start(self):
        """システムを起動"""
        print("=" * 60)
        print("モジュール化自動運転システム")
        print("=" * 60)
        print("特徴:")
        print("- モジュール化されたモデル構造")
        print("- 簡単にモデルを交換可能")
        print("- リアルタイム推論結果表示")
        print("- ESCキーで終了")
        print("-" * 60)
        
        # カメラ初期化
        if not self._initialize_camera():
            return
        
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nシステムを停止します...")
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
        finally:
            self._cleanup()
    
    def _initialize_camera(self) -> bool:
        """カメラを初期化"""
        print("カメラを初期化中...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"エラー: カメラID {self.camera_id} を開けません")
            print("解決策:")
            print("1. macOS設定 > プライバシー > カメラで権限を許可")
            print("2. 他のカメラアプリを終了")
            return False
        
        # カメラ設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"カメラ初期化完了: {actual_width}x{actual_height} @ {actual_fps}fps")
        return True
    
    def _main_loop(self):
        """メイン処理ループ"""
        while self.running:
            loop_start = time.time()
            
            # フレーム取得
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("フレーム取得エラー")
                continue
            
            self.frame_count += 1
            
            # モデル推論
            lane_result = self.lane_model.detect(frame)
            objects = self.object_model.detect(frame)
            command = self.decision_model.decide(lane_result, objects)
            
            # FPS更新
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # 可視化
            self._visualize(frame, lane_result, objects, command)
            
            # フレームレート制御
            process_time = time.time() - loop_start
            if process_time < 1/30:  # 30 FPS目標
                time.sleep(1/30 - process_time)
    
    def _visualize(self, frame: np.ndarray, lane_result: LaneResult, 
                  objects: list[DetectionResult], command: DrivingCommand):
        """推論結果を可視化"""
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # 背景パネル
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # 車線検出結果
        self._draw_lanes(vis_frame, lane_result)
        
        # 物体検出結果
        self._draw_objects(vis_frame, objects)
        
        # 運転指示
        self._draw_command(vis_frame, command, width)
        
        # システム情報
        self._draw_system_info(vis_frame, width)
        
        # 緊急警告
        if command.urgency >= 8:
            self._draw_emergency_warning(vis_frame, width, height)
        
        # 制御ヒント
        cv2.putText(vis_frame, "ESC: Exit | Models: Lane/Obj/Decision", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # 表示
        cv2.imshow('Modular Autonomous Driving System', vis_frame)
        
        # ESCキーで終了
        if cv2.waitKey(1) & 0xFF == 27:
            self.running = False
    
    def _draw_lanes(self, vis_frame: np.ndarray, lane_result: LaneResult):
        """車線検出結果を描画"""
        lane_color = (0, 255, 0) if lane_result.left_lane and lane_result.right_lane else (0, 165, 255)
        
        # 車線描画
        if lane_result.left_lane:
            x1, y1, x2, y2 = lane_result.left_lane
            cv2.line(vis_frame, (x1, y1), (x2, y2), lane_color, 3)
            cv2.circle(vis_frame, (x1, y1), 5, (255, 255, 0), -1)
        
        if lane_result.right_lane:
            x1, y1, x2, y2 = lane_result.right_lane
            cv2.line(vis_frame, (x1, y1), (x2, y2), lane_color, 3)
            cv2.circle(vis_frame, (x1, y1), 5, (255, 255, 0), -1)
        
        # 中心線
        if lane_result.lane_center:
            height = vis_frame.shape[0]
            cv2.line(vis_frame, (lane_result.lane_center, height), 
                    (lane_result.lane_center, height - 80), (255, 255, 255), 2)
        
        # 状態表示
        status = "DETECTED" if lane_result.left_lane and lane_result.right_lane else "DETECTING"
        cv2.putText(vis_frame, f"Lanes: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)
    
    def _draw_objects(self, vis_frame: np.ndarray, objects: list[DetectionResult]):
        """物体検出結果を描画"""
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj.bbox
            
            # 距離に応じた色
            if obj.distance < 5.0:
                color = (0, 0, 255)  # 赤：危険
                warning = "!"
            elif obj.distance < 15.0:
                color = (0, 165, 255)  # オレンジ：注意
                warning = ""
            else:
                color = (0, 255, 0)  # 緑：安全
                warning = ""
            
            # バウンディングボックス
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # ラベル
            model_type = obj.additional_info.get('model', 'unknown')[:3].upper()
            label = f"{obj.class_name}({model_type}): {obj.distance:.1f}m{warning}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 番号
            cv2.putText(vis_frame, f"#{i+1}", (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 物体数表示
        obj_color = (0, 255, 0) if objects else (255, 255, 0)
        cv2.putText(vis_frame, f"Objects: {len(objects)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, obj_color, 2)
    
    def _draw_command(self, vis_frame: np.ndarray, command: DrivingCommand, width: int):
        """運転指示を描画"""
        # 背景色
        if command.urgency >= 8:
            bg_color = (0, 0, 255)  # 赤：緊急
        elif command.urgency >= 5:
            bg_color = (0, 100, 200)  # オレンジ：注意
        else:
            bg_color = (0, 150, 0)  # 緑：正常
        
        # 指示パネル
        panel_x = width - 280
        cv2.rectangle(vis_frame, (panel_x, 10), (width - 10, 110), bg_color, -1)
        cv2.rectangle(vis_frame, (panel_x, 10), (width - 10, 110), (255, 255, 255), 2)
        
        # アクション
        cv2.putText(vis_frame, command.action, (panel_x + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # 詳細情報
        cv2.putText(vis_frame, f"Conf: {command.confidence:.2f}", (panel_x + 10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Urgency: {command.urgency}/10", (panel_x + 10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_system_info(self, vis_frame: np.ndarray, width: int):
        """システム情報を描画"""
        # FPS
        fps_color = (0, 255, 0) if self.fps >= 25 else (0, 255, 255) if self.fps >= 15 else (0, 0, 255)
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # モデル統計
        lane_stats = self.lane_model.get_stats()
        obj_stats = self.object_model.get_stats()
        decision_stats = self.decision_model.get_stats()
        
        stats_text = f"L:{lane_stats['fps']:.0f} O:{obj_stats['fps']:.0f} D:{decision_stats['fps']:.0f}"
        cv2.putText(vis_frame, stats_text, (200, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 実行時間
        runtime = time.time() - self.start_time
        cv2.putText(vis_frame, f"Time: {runtime:.0f}s", (400, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_emergency_warning(self, vis_frame: np.ndarray, width: int, height: int):
        """緊急警告を描画"""
        if int(time.time() * 3) % 2 == 0:  # 点滅効果
            cv2.rectangle(vis_frame, (0, 0), (width, height), (0, 0, 255), 10)
            cv2.putText(vis_frame, "EMERGENCY!", (width//2 - 100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
    
    def _cleanup(self):
        """クリーンアップ処理"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("システムを停止しました")

def main():
    """メイン関数"""
    # 設定 - ここでモデルを切り替え
    config = {
        'camera_id': 0,
        'lane_detection': {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50
        },
        'object_detection': {
            'model_type': 'contour',  # 'contour', 'yolo', 'custom'
            'confidence_threshold': 0.5
        },
        'decision_making': {
            'emergency_distance': 5.0,
            'safe_distance': 15.0
        }
    }
    
    system = AutonomousDrivingSystem(config)
    system.start()

if __name__ == "__main__":
    main()
