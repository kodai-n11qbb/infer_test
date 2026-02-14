#!/usr/bin/env python3
"""
シンプルなリアルタイム自動運転デモ
推論結果を確実に可視化する
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class DetectionResult:
    """検出結果データクラス"""
    lanes_detected: bool
    objects_count: int
    action: str
    confidence: float
    urgency: int

class SimpleLaneDetector:
    """シンプルな車線検出 - エッジ検出ベース"""
    
    def __init__(self):
        self.frame_count = 0
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List]:
        """
        車線を検出 - シンプルなエッジ検出
        戻り値: (検出成功フラグ, 車線座標リスト)
        """
        self.frame_count += 1
        
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred, 50, 150)
        
        # ハフ変換で線分検出
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=40, maxLineGap=20)
        
        if lines is None:
            return False, []
        
        # 線分を左右に分類
        left_lines = []
        right_lines = []
        height, width = frame.shape[:2]
        center_x = width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # ゼロ除算回避
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # 傾きで左右を判定
            if slope < -0.5:  # 左車線
                left_lines.append(line[0])
            elif slope > 0.5:  # 右車線
                right_lines.append(line[0])
        
        # 簡単な検出判定
        detected = len(left_lines) > 0 and len(right_lines) > 0
        
        return detected, left_lines + right_lines

class SimpleObjectDetector:
    """シンプルな物体検出 - 輪郭検出ベース"""
    
    def __init__(self):
        self.frame_count = 0
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        物体を検出 - 輪郭検出ベースの簡易実装
        実際のYOLOの代わりに色と輪郭で物体を検出
        """
        self.frame_count += 1
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 赤色と青色の範囲を定義（車両を模倣）
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # 色マスク作成
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 統合マスク
        mask = cv2.bitwise_or(mask_red, mask_blue)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 500:  # 小さなノイズを無視
                continue
                
            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(contour)
            
            # 簡易距離推定（サイズベース）
            distance = max(2.0, min(50.0, 10000 / (w * h)))
            
            # 色でクラス判定
            center_x = x + w // 2
            center_y = y + h // 2
            pixel_color = hsv[center_y, center_x]
            
            if pixel_color[0] < 10 or pixel_color[0] > 170:  # 赤色
                class_name = "car"
            else:  # 青色
                class_name = "truck"
            
            objects.append({
                'class': class_name,
                'bbox': [x, y, x + w, y + h],
                'confidence': min(0.9, area / 10000),
                'distance': distance
            })
        
        return objects

class SimpleDecisionMaker:
    """シンプルな意思決定 - ルールベース"""
    
    def __init__(self):
        self.action_history = []
    
    def decide(self, lanes_detected: bool, objects: List[dict]) -> DetectionResult:
        """
        運転指示を決定 - シンプルなルールベース
        """
        # 緊急事態チェック
        for obj in objects:
            if obj['distance'] < 5.0:
                return DetectionResult(
                    lanes_detected=lanes_detected,
                    objects_count=len(objects),
                    action="BRAKE",
                    confidence=0.95,
                    urgency=10
                )
        
        # 前方障害物チェック
        front_objects = [obj for obj in objects if obj['distance'] < 15.0]
        if front_objects:
            return DetectionResult(
                lanes_detected=lanes_detected,
                objects_count=len(objects),
                action="SLOW_DOWN",
                confidence=0.7,
                urgency=5
            )
        
        # 車線状態チェック
        if not lanes_detected:
            return DetectionResult(
                lanes_detected=False,
                objects_count=len(objects),
                action="STRAIGHT",
                confidence=0.5,
                urgency=2
            )
        
        # 正常状態
        return DetectionResult(
            lanes_detected=True,
            objects_count=len(objects),
            action="ACCELERATE",
            confidence=0.8,
            urgency=1
        )

class SimpleAutonomousDriver:
    """シンプルな自動運転システム - 確実な可視化重視"""
    
    def __init__(self):
        self.lane_detector = SimpleLaneDetector()
        self.object_detector = SimpleObjectDetector()
        self.decision_maker = SimpleDecisionMaker()
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # カメラ設定
        self.camera_id = 0
        self.cap = None
    
    def start(self):
        """システムを起動"""
        print("=" * 60)
        print("シンプル自動運転システム - 推論結果可視化デモ")
        print("=" * 60)
        print("特徴:")
        print("- シンプルなアルゴリズムで確実な動作")
        print("- リアルタイム推論結果の詳細表示")
        print("- ESCキーで終了")
        print("-" * 60)
        
        # カメラ初期化
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("エラー: カメラを開けません")
            print("解決策:")
            print("1. macOS設定 > プライバシー > カメラで権限を許可")
            print("2. 他のカメラアプリを終了")
            return
        
        # カメラ設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("カメラ初期化完了 - リアルタイム処理を開始")
        self.running = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\nシステムを停止します...")
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
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
            
            # 推論処理
            lanes_detected, lane_lines = self.lane_detector.detect(frame)
            objects = self.object_detector.detect(frame)
            decision = self.decision_maker.decide(lanes_detected, objects)
            
            # FPS更新
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # 可視化
            self._visualize(frame, lanes_detected, lane_lines, objects, decision)
            
            # フレームレート制御
            process_time = time.time() - loop_start
            if process_time < 1/30:  # 30 FPS目標
                time.sleep(1/30 - process_time)
    
    def _visualize(self, frame: np.ndarray, lanes_detected: bool, 
                  lane_lines: List, objects: List[dict], decision: DetectionResult):
        """
        推論結果を詳細に可視化
        """
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # === 背景情報パネル ===
        # 上部に半透明黒パネル
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # === 車線検出結果 ===
        # 検出された線分を描画
        for line in lane_lines:
            x1, y1, x2, y2 = line
            cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 車線検出状態表示
        lane_status = "DETECTED" if lanes_detected else "NOT DETECTED"
        lane_color = (0, 255, 0) if lanes_detected else (0, 0, 255)
        cv2.putText(vis_frame, f"Lanes: {lane_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)
        
        # === 物体検出結果 ===
        for i, obj in enumerate(objects):
            x1, y1, x2, y2 = obj['bbox']
            
            # 距離に応じた色
            if obj['distance'] < 5.0:
                color = (0, 0, 255)  # 赤：危険
                label_suffix = "!"
            elif obj['distance'] < 15.0:
                color = (0, 165, 255)  # オレンジ：注意
                label_suffix = ""
            else:
                color = (0, 255, 0)  # 緑：安全
                label_suffix = ""
            
            # バウンディングボックス
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # ラベル
            label = f"{obj['class']}: {obj['distance']:.1f}m{label_suffix}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 物体番号
            cv2.putText(vis_frame, f"#{i+1}", (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 物体検出数表示
        obj_color = (0, 255, 0) if objects else (255, 255, 0)
        cv2.putText(vis_frame, f"Objects: {len(objects)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, obj_color, 2)
        
        # === 運転指示表示 ===
        # 指示に応じた背景色
        if decision.urgency >= 8:
            bg_color = (0, 0, 255)  # 赤：緊急
        elif decision.urgency >= 5:
            bg_color = (0, 100, 200)  # オレンジ：注意
        else:
            bg_color = (0, 150, 0)  # 緑：正常
        
        # 指示パネル
        panel_x = width - 250
        cv2.rectangle(vis_frame, (panel_x, 10), (width - 10, 110), bg_color, -1)
        cv2.rectangle(vis_frame, (panel_x, 10), (width - 10, 110), (255, 255, 255), 2)
        
        # アクション表示（大きく）
        cv2.putText(vis_frame, decision.action, (panel_x + 10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # 信頼度と緊急度
        cv2.putText(vis_frame, f"Conf: {decision.confidence:.2f}", (panel_x + 10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Urgency: {decision.urgency}/10", (panel_x + 10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === 性能情報 ===
        # FPS表示
        fps_color = (0, 255, 0) if self.fps >= 25 else (0, 255, 255) if self.fps >= 15 else (0, 0, 255)
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # フレームカウント
        cv2.putText(vis_frame, f"Frame: {self.frame_count}", (150, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 実行時間
        runtime = time.time() - self.start_time
        cv2.putText(vis_frame, f"Time: {runtime:.0f}s", (300, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # === 緊急警告 ===
        if decision.urgency >= 8:
            # 点滅効果
            if int(time.time() * 3) % 2 == 0:
                cv2.rectangle(vis_frame, (0, 0), (width, height), (0, 0, 255), 10)
                cv2.putText(vis_frame, "EMERGENCY!", (width//2 - 80, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        
        # === 制御ヒント ===
        cv2.putText(vis_frame, "ESC: Exit", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # ウィンドウに表示
        cv2.imshow('Autonomous Driving - Real-time Inference', vis_frame)
        
        # ESCキーで終了
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.running = False

def main():
    """メイン関数"""
    driver = SimpleAutonomousDriver()
    driver.start()

if __name__ == "__main__":
    main()
