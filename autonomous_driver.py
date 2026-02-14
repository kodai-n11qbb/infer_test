#!/usr/bin/env python3
"""
リアルタイム自動運転システム
画面から運転環境を解析し、適切な運転指示を生成する
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mss
import pyautogui
import time
from collections import deque
import threading
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math

@dataclass
class DrivingInstruction:
    """運転指示データクラス"""
    action: str  # 'accelerate', 'brake', 'turn_left', 'turn_right', 'straight'
    confidence: float  # 指示の信頼度 (0.0-1.0)
    urgency: int  # 緊急度 (1-10, 10が最も緊急)
    reason: str  # 指示の理由

class LaneDetector:
    """車線検出クラス - Cannyエッジ検出とハフ変換を使用"""
    
    def __init__(self):
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frame_history = deque(maxlen=5)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレームの前処理 - グレースケール変換と領域抽出"""
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred, 50, 150)
        
        # 関心領域（ROI）マスク - 画面下部の道路領域に焦点
        height, width = edges.shape
        roi_vertices = np.array([
            [(0, height), (width * 0.45, height * 0.6), 
             (width * 0.55, height * 0.6), (width, height)]
        ], dtype=np.int32)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges
    
    def detect_lanes(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        車線を検出し、左右の車線座標を返す
        戻り値: (left_lane, right_lane) - 検出できない場合はNone
        """
        processed = self.preprocess_frame(frame)
        
        # ハフ変換で線分検出
        lines = cv2.HoughLinesP(processed, 1, np.pi/180, 50, 
                               minLineLength=40, maxLineGap=20)
        
        if lines is None:
            return None, None
        
        left_lines = []
        right_lines = []
        height, width = frame.shape[:2]
        center_x = width / 2
        
        # 線分を左右に分類
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # ゼロ除算回避
            
            # 傾きで左右を判定
            if slope < -0.5:  # 左車線（負の傾き）
                left_lines.append(line[0])
            elif slope > 0.5:  # 右車線（正の傾き）
                right_lines.append(line[0])
        
        # 車線を平均化して一本の線に
        left_lane = self._average_lines(left_lines, height, width) if left_lines else None
        right_lane = self._average_lines(right_lines, height, width) if right_lines else None
        
        return left_lane, right_lane
    
    def _average_lines(self, lines: List[np.ndarray], height: int, width: int) -> np.ndarray:
        """複数の線分を平均化して一本の車線を生成"""
        if not lines:
            return np.array([])
        
        # 線分の傾きと切片を計算
        slopes = []
        intercepts = []
        
        for x1, y1, x2, y2 in lines:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
        
        # 平均値を計算
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        # 車線の両端の座標を計算
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - avg_intercept) / (avg_slope + 1e-6))
        x2 = int((y2 - avg_intercept) / (avg_slope + 1e-6))
        
        return np.array([x1, y1, x2, y2])

class ObjectDetector:
    """物体検出クラス - YOLOv8を使用"""
    
    def __init__(self):
        # YOLOv8モデルをロード（事前学習済みモデル）
        self.model = YOLO('yolov8n.pt')  # 軽量モデルを選択
        self.target_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
    
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """
        フレーム内の物体を検出
        戻り値: 検出された物体のリスト [{class, bbox, confidence, distance}, ...]
        """
        results = self.model(frame, verbose=False)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # クラスと信頼度を取得
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    # ターゲットクラスのみを処理
                    if class_name in self.target_classes and confidence > 0.5:
                        # バウンディングボックス座標
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 距離を推定（バウンディングボックスのサイズに基づく簡易推定）
                        distance = self._estimate_distance(x2 - x1, y2 - y1, class_name)
                        
                        detected_objects.append({
                            'class': class_name,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'distance': distance
                        })
        
        return detected_objects
    
    def _estimate_distance(self, width: float, height: float, class_name: str) -> float:
        """
        バウンディングボックスサイズから距離を推定
        簡易的な距離推定（実際のシステムではカメラキャリブレーションが必要）
        """
        # クラスごとの基準サイズ（ピクセル）
        reference_sizes = {
            'car': 100,
            'truck': 150,
            'bus': 180,
            'motorcycle': 60,
            'bicycle': 50,
            'person': 40
        }
        
        ref_size = reference_sizes.get(class_name, 100)
        # サイズ比から距離を逆算（簡易計算）
        estimated_distance = ref_size / max(width, height) * 10  # メートル単位
        
        return max(1.0, min(estimated_distance, 100.0))  # 1m〜100mの範囲に制限

class DrivingDecisionMaker:
    """運転意思決定クラス - 検出結果から運転指示を生成"""
    
    def __init__(self):
        self.instruction_history = deque(maxlen=10)
        self.emergency_brake_threshold = 5.0  # 緊急ブレーキ距離（メートル）
        self.safe_following_distance = 15.0  # 安全な車間距離（メートル）
    
    def make_decision(self, left_lane: Optional[np.ndarray], 
                     right_lane: Optional[np.ndarray], 
                     objects: List[dict]) -> DrivingInstruction:
        """
        車線情報と物体検出結果から運転指示を生成
        """
        # 緊急事態のチェック
        emergency_instruction = self._check_emergency(objects)
        if emergency_instruction:
            return emergency_instruction
        
        # 車線維持のチェック
        lane_instruction = self._analyze_lane_position(left_lane, right_lane)
        
        # 前方障害物のチェック
        obstacle_instruction = self._check_front_obstacles(objects)
        
        # 最適な指示を選択
        instructions = [inst for inst in [lane_instruction, obstacle_instruction] if inst]
        
        if instructions:
            # 緊急度が高い指示を優先
            best_instruction = max(instructions, key=lambda x: (x.urgency, x.confidence))
        else:
            # デフォルト：直進
            best_instruction = DrivingInstruction(
                action='straight',
                confidence=0.8,
                urgency=1,
                reason='正常な直進状態'
            )
        
        self.instruction_history.append(best_instruction)
        return best_instruction
    
    def _check_emergency(self, objects: List[dict]) -> Optional[DrivingInstruction]:
        """緊急事態をチェック"""
        for obj in objects:
            if obj['distance'] < self.emergency_brake_threshold:
                return DrivingInstruction(
                    action='brake',
                    confidence=0.95,
                    urgency=10,
                    reason=f'緊急：{obj["class"]}が{obj["distance"]:.1f}mに接近'
                )
        return None
    
    def _analyze_lane_position(self, left_lane: Optional[np.ndarray], 
                            right_lane: Optional[np.ndarray]) -> Optional[DrivingInstruction]:
        """車線位置を解析して操舵指示を生成"""
        height = 480  # 仮のフレーム高さ
        
        if left_lane is None and right_lane is None:
            return DrivingInstruction(
                action='straight',
                confidence=0.3,
                urgency=2,
                reason='車線が検出できないため直進維持'
            )
        
        # 車線の中心を計算
        lane_center = None
        if left_lane is not None and right_lane is not None:
            lane_center = (left_lane[0] + right_lane[0]) / 2
        elif left_lane is not None:
            lane_center = left_lane[0] + 100  # 左車線のみの場合
        elif right_lane is not None:
            lane_center = right_lane[0] - 100  # 右車線のみの場合
        
        if lane_center is not None:
            frame_center = 640 / 2  # 仮のフレーム幅の中心
            deviation = lane_center - frame_center
            
            # 逸脱量に応じて操舵指示
            if abs(deviation) > 50:
                if deviation > 0:
                    return DrivingInstruction(
                        action='turn_right',
                        confidence=0.8,
                        urgency=5,
                        reason=f'右に逸脱中（偏差:{deviation:.1f}px）'
                    )
                else:
                    return DrivingInstruction(
                        action='turn_left',
                        confidence=0.8,
                        urgency=5,
                        reason=f'左に逸脱中（偏差:{deviation:.1f}px）'
                    )
        
        return None
    
    def _check_front_obstacles(self, objects: List[dict]) -> Optional[DrivingInstruction]:
        """前方障害物をチェック"""
        front_objects = []
        frame_center_x = 640 / 2
        frame_width = 640
        
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            obj_center_x = (x1 + x2) / 2
            
            # 前方領域にある物体を判定
            if abs(obj_center_x - frame_center_x) < frame_width * 0.3:
                front_objects.append(obj)
        
        if front_objects:
            # 最も近い前方物体を取得
            closest_obj = min(front_objects, key=lambda x: x['distance'])
            
            if closest_obj['distance'] < self.safe_following_distance:
                if closest_obj['distance'] < self.emergency_brake_threshold:
                    return DrivingInstruction(
                        action='brake',
                        confidence=0.9,
                        urgency=8,
                        reason=f'前方{closest_obj["class"]}が危険距離：{closest_obj["distance"]:.1f}m'
                    )
                else:
                    return DrivingInstruction(
                        action='brake',
                        confidence=0.6,
                        urgency=4,
                        reason=f'前方{closest_obj["class"]}に減速：{closest_obj["distance"]:.1f}m'
                    )
        
        return None

class AutonomousDrivingSystem:
    """自動運転システムのメインクラス"""
    
    def __init__(self):
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        self.decision_maker = DrivingDecisionMaker()
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # スクリーンキャプチャの設定
        self.monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
    
    def start(self):
        """システムを起動"""
        print("自動運転システムを起動中...")
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nシステムを停止します...")
        finally:
            self.running = False
    
    def _main_loop(self):
        """メイン処理ループ"""
        with mss.mss() as sct:
            while self.running:
                loop_start = time.time()
                
                # スクリーンキャプチャ
                frame = self._capture_screen(sct)
                if frame is None:
                    continue
                
                # 車線検出
                left_lane, right_lane = self.lane_detector.detect_lanes(frame)
                
                # 物体検出
                objects = self.object_detector.detect_objects(frame)
                
                # 意思決定
                instruction = self.decision_maker.make_decision(left_lane, right_lane, objects)
                
                # 結果の表示
                self._visualize_results(frame, left_lane, right_lane, objects, instruction)
                
                # FPS計算
                self._update_fps()
                
                # フレームレート制御（目標: 30 FPS）
                elapsed = time.time() - loop_start
                if elapsed < 1/30:
                    time.sleep(1/30 - elapsed)
    
    def _capture_screen(self, sct) -> Optional[np.ndarray]:
        """スクリーンからフレームをキャプチャ"""
        try:
            sct_img = sct.grab(self.monitor)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"スクリーンキャプチャエラー: {e}")
            return None
    
    def _visualize_results(self, frame: np.ndarray, left_lane: Optional[np.ndarray], 
                          right_lane: Optional[np.ndarray], objects: List[dict], 
                          instruction: DrivingInstruction):
        """検出結果と指示を可視化"""
        vis_frame = frame.copy()
        
        # 車線の描画
        if left_lane is not None:
            cv2.line(vis_frame, (left_lane[0], left_lane[1]), 
                    (left_lane[2], left_lane[3]), (0, 255, 0), 3)
        
        if right_lane is not None:
            cv2.line(vis_frame, (right_lane[0], right_lane[1]), 
                    (right_lane[2], right_lane[3]), (0, 255, 0), 3)
        
        # 物体の描画
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # ラベルと距離を表示
            label = f"{obj['class']}: {obj['distance']:.1f}m"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 運転指示の表示
        instruction_text = f"Action: {instruction.action}"
        confidence_text = f"Confidence: {instruction.confidence:.2f}"
        urgency_text = f"Urgency: {instruction.urgency}/10"
        reason_text = f"Reason: {instruction.reason}"
        
        # 指示に応じた色を設定
        if instruction.urgency >= 8:
            color = (0, 0, 255)  # 赤：緊急
        elif instruction.urgency >= 5:
            color = (0, 165, 255)  # オレンジ：注意
        else:
            color = (0, 255, 0)  # 緑：正常
        
        cv2.putText(vis_frame, instruction_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis_frame, confidence_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_frame, urgency_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_frame, reason_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS表示
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(vis_frame, fps_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 結果を表示
        cv2.imshow('Autonomous Driving System', vis_frame)
        
        # ESCキーで終了
        if cv2.waitKey(1) & 0xFF == 27:
            self.running = False
    
    def _update_fps(self):
        """FPSを更新"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

def main():
    """メイン関数"""
    print("=" * 50)
    print("リアルタイム自動運転システム")
    print("画面から運転環境を解析し、適切な指示を生成")
    print("=" * 50)
    print("\n制御:")
    print("- ESCキー: システム停止")
    print("- ウィンドウを閉じる: システム停止")
    print("\n起動中...")
    
    # システムの起動
    system = AutonomousDrivingSystem()
    system.start()
    
    cv2.destroyAllWindows()
    print("システムを停止しました")

if __name__ == "__main__":
    main()
