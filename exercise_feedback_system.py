# -*- coding: utf-8 -*-
"""
통합 운동 피드백 시스템
- BiLSTM으로 운동 종류 분류
- 자세 분석 및 LLM 피드백 제공
"""

import os
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

# 서브모듈 import
from models.bilstm_classifier import ExerciseClassifier
from feedback.pose_analyzer import PoseAnalyzer
from feedback.llm_feedback import FeedbackGenerator


class ExerciseFeedbackSystem:
    """통합 운동 피드백 시스템"""
    
    def __init__(self, 
                 model_path: str,
                 label_encoder_path: str,
                 openai_api_key: str,
                 model_type: str = "gpt"):
        """
        Args:
            model_path: BiLSTM 모델 경로
            label_encoder_path: Label encoder 경로
            openai_api_key: OpenAI API 키
        """
        # BiLSTM 분류기 초기화
        self.classifier = ExerciseClassifier(model_path, label_encoder_path)
        
        # 자세 분석기 초기화
        self.pose_analyzer = PoseAnalyzer()
        
        # LLM 피드백 생성기 초기화
        self.feedback_generator = FeedbackGenerator(openai_api_key, model_type)
    
    def process_video(self,
                     user_video_path: str,
                     gt_video_path: Optional[str] = None,
                     output_dir: str = "./output") -> Dict:
        """
        비디오 처리 및 피드백 생성
        
        Args:
            user_video_path: 사용자 운동 영상 경로
            gt_video_path: Ground truth 영상 경로 (없으면 None)
            output_dir: 출력 디렉토리
            
        Returns:
            {
                'exercise_type': str,
                'confidence': float,
                'feedback_video': str,
                'feedback_text': str,
                'json_summary': dict
            }
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("[1/4] 관절 데이터 추출 중...")
        # 1. 관절 데이터 추출
        pose_data = self.pose_analyzer.extract_landmarks(user_video_path)
        
        print("[2/4] 운동 종류 분류 중...")
        # 2. BiLSTM으로 운동 종류 분류
        exercise_type, confidence = self.classifier.predict(pose_data)
        print(f"  -> 분류 결과: {exercise_type} (신뢰도: {confidence:.2%})")
        
        # 3. GT 영상 선택 (없으면 기본 GT 사용)
        if gt_video_path is None:
            gt_video_path = self._get_default_gt(exercise_type)
        
        print("[3/4] 자세 분석 및 시각화 중...")
        print(f"  -> GT 영상: {gt_video_path}")
        # 4. 자세 비교 및 시각화
        feedback_video_path = os.path.join(output_dir, "feedback_video.mp4")
        full_json_path = os.path.join(output_dir, "feedback_full.json")
        summary_json_path = os.path.join(output_dir, "feedback_summary.json")
        
        self.pose_analyzer.evaluate(
            gt_video=gt_video_path,
            usr_video=user_video_path,
            save_path=feedback_video_path,
            full_json_path=full_json_path,
            summary_json_path=summary_json_path
        )
        
        print("[4/4] LLM 피드백 생성 중...")
        # 5. LLM 피드백 생성
        import json
        with open(summary_json_path, 'r') as f:
            summary_data = json.load(f)
        
        feedback_text = self.feedback_generator.generate_feedback(
            exercise_type=exercise_type,
            summary_data=summary_data,
            orientation="front"  # 또는 "side"
        )
        
        print("\n✅ 처리 완료!")
        return {
            'exercise_type': exercise_type,
            'confidence': confidence,
            'feedback_video': feedback_video_path,
            'feedback_text': feedback_text,
            'json_summary': summary_data
        }
    
    def _get_default_gt(self, exercise_type: str) -> str:
        """운동 종류에 맞는 기본 GT 영상 경로 반환"""
        gt_mapping = {
            "squat": "./data/gt_videos/squat_gt.mp4",
            "lunge": "./data/gt_videos/lunge_gt.mp4",
            "side_lunge": "./data/gt_videos/side_lunge_gt.mp4",
            "situp": "./data/gt_videos/situp_gt.mp4",
            "highknees": "./data/gt_videos/highknees_gt.mp4",
            "bridge": "./data/gt_videos/bridge_gt.mp4",
            "cobra_pose": "./data/gt_videos/cobra_pose_gt.mp4",
            "jumping_jack": "./data/gt_videos/jumpingjack_gt.mp4",
        }
        return gt_mapping.get(exercise_type, "./data/gt_videos/default_gt.mp4")


# ============= 사용 예시 =============
if __name__ == "__main__":
    # 시스템 초기화
    system = ExerciseFeedbackSystem(
        model_path="./models/model.pt",
        label_encoder_path="./models/label_encoder.pkl",
        openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    )
    
    # 비디오 처리
    result = system.process_video(
        user_video_path="./data/user_videos/user_exercise.mp4",
        gt_video_path=None,  # None이면 자동으로 선택
        output_dir="./output"
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"운동 종류: {result['exercise_type']}")
    print(f"신뢰도: {result['confidence']:.2%}")
    print(f"피드백 영상: {result['feedback_video']}")
    print("\n[피드백 내용]")
    print(result['feedback_text'])
    print("="*60)
