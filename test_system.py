#!/usr/bin/env python3
"""
自動運転システムテストモジュール
各コンポーネントの動作検証用
"""

import cv2
import numpy as np
import time
from autonomous_driver import LaneDetector, ObjectDetector, DrivingDecisionMaker, DrivingInstruction

def create_test_frame(frame_type="normal"):
    """テスト用のダミーフレームを生成"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    if frame_type == "normal":
        # 道路のベース
        frame[200:, :] = [80, 80, 80]  # グレーの道路
        
        # 車線
        cv2.line(frame, (200, 480), (280, 240), (255, 255, 255), 5)
        cv2.line(frame, (440, 480), (360, 240), (255, 255, 255), 5)
        
        # 前方車両
        cv2.rectangle(frame, (270, 180), (370, 250), (0, 0, 255), -1)
        
    elif frame_type == "emergency":
        # 緊急状況
        frame[200:, :] = [80, 80, 80]
        cv2.line(frame, (200, 480), (280, 240), (255, 255, 255), 5)
        cv2.line(frame, (440, 480), (360, 240), (255, 255, 255), 5)
        # 非常に近い障害物
        cv2.rectangle(frame, (290, 300), (350, 400), (0, 0, 255), -1)
        
    elif frame_type == "lane_deviation":
        # 車線逸脱
        frame[200:, :] = [80, 80, 80]
        cv2.line(frame, (150, 480), (230, 240), (255, 255, 255), 5)
        cv2.line(frame, (390, 480), (310, 240), (255, 255, 255), 5)
        
    return frame

def test_lane_detector():
    """車線検出器のテスト"""
    print("車線検出器テスト中...")
    detector = LaneDetector()
    
    # 正常なフレームでテスト
    frame = create_test_frame("normal")
    left_lane, right_lane = detector.detect_lanes(frame)
    
    print(f"左車線検出: {left_lane is not None}")
    print(f"右車線検出: {right_lane is not None}")
    
    if left_lane is not None:
        print(f"左車線座標: {left_lane}")
    if right_lane is not None:
        print(f"右車線座標: {right_lane}")
    
    return left_lane is not None and right_lane is not None

def test_object_detector():
    """物体検出器のテスト"""
    print("\n物体検出器テスト中...")
    detector = ObjectDetector()
    
    # 正常なフレームでテスト
    frame = create_test_frame("normal")
    objects = detector.detect_objects(frame)
    
    print(f"検出物体数: {len(objects)}")
    for obj in objects:
        print(f"- {obj['class']}: 信頼度{obj['confidence']:.2f}, 距離{obj['distance']:.1f}m")
    
    return len(objects) > 0

def test_decision_maker():
    """意思決定器のテスト"""
    print("\n意思決定器テスト中...")
    decision_maker = DrivingDecisionMaker()
    
    # 正常状況
    frame = create_test_frame("normal")
    detector = LaneDetector()
    obj_detector = ObjectDetector()
    
    left_lane, right_lane = detector.detect_lanes(frame)
    objects = obj_detector.detect_objects(frame)
    
    instruction = decision_maker.make_decision(left_lane, right_lane, objects)
    print(f"通常時指示: {instruction.action} (緊急度: {instruction.urgency})")
    
    # 緊急状況
    emergency_frame = create_test_frame("emergency")
    left_lane, right_lane = detector.detect_lanes(emergency_frame)
    objects = obj_detector.detect_objects(emergency_frame)
    
    emergency_instruction = decision_maker.make_decision(left_lane, right_lane, objects)
    print(f"緊急時指示: {emergency_instruction.action} (緊急度: {emergency_instruction.urgency})")
    
    return instruction.action == "straight" and emergency_instruction.action == "brake"

def test_performance():
    """性能テスト"""
    print("\n性能テスト中...")
    
    detector = LaneDetector()
    obj_detector = ObjectDetector()
    decision_maker = DrivingDecisionMaker()
    
    frame = create_test_frame("normal")
    
    # 100回の処理時間を計測
    start_time = time.time()
    
    for _ in range(100):
        left_lane, right_lane = detector.detect_lanes(frame)
        objects = obj_detector.detect_objects(frame)
        instruction = decision_maker.make_decision(left_lane, right_lane, objects)
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / 100
    fps = 1 / avg_time
    
    print(f"平均処理時間: {avg_time*1000:.2f}ms")
    print(f"理論上のFPS: {fps:.1f}")
    
    return fps > 10  # 最低10FPSを目標

def visualize_test():
    """可視化テスト"""
    print("\n可視化テスト中...")
    print("テスト画像を表示します。何かキーを押すと次の画像に進みます。")
    
    test_frames = [
        ("正常状況", create_test_frame("normal")),
        ("緊急状況", create_test_frame("emergency")),
        ("車線逸脱", create_test_frame("lane_deviation"))
    ]
    
    detector = LaneDetector()
    obj_detector = ObjectDetector()
    decision_maker = DrivingDecisionMaker()
    
    for name, frame in test_frames:
        # 検出処理
        left_lane, right_lane = detector.detect_lanes(frame)
        objects = obj_detector.detect_objects(frame)
        instruction = decision_maker.make_decision(left_lane, right_lane, objects)
        
        # 可視化
        vis_frame = frame.copy()
        
        # 車線描画
        if left_lane is not None:
            cv2.line(vis_frame, (left_lane[0], left_lane[1]), 
                    (left_lane[2], left_lane[3]), (0, 255, 0), 3)
        if right_lane is not None:
            cv2.line(vis_frame, (right_lane[0], right_lane[1]), 
                    (right_lane[2], right_lane[3]), (0, 255, 0), 3)
        
        # 物体描画
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{obj['class']}: {obj['distance']:.1f}m"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 指示表示
        text = f"{instruction.action} (緊急度: {instruction.urgency})"
        cv2.putText(vis_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(f"テスト - {name}", vis_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("可視化テスト完了")

def main():
    """メインテスト関数"""
    print("=" * 50)
    print("自動運転システム コンポーネントテスト")
    print("=" * 50)
    
    tests = [
        ("車線検出", test_lane_detector),
        ("物体検出", test_object_detector),
        ("意思決定", test_decision_maker),
        ("性能", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}テスト: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"✗ {test_name}テスト: ERROR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n合計: {passed}/{total} テスト通過")
    
    # 可視化テスト（オプション）
    if passed >= 3:  # 主要テストが通過した場合のみ
        print("\n可視化テストを実行しますか？ (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            visualize_test()
    
    print("\nテスト完了")

if __name__ == "__main__":
    main()
