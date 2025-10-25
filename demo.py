#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 운동 피드백 시스템 데모
"""

import os
import argparse
from pathlib import Path
from exercise_feedback_system import ExerciseFeedbackSystem


def main():
    parser = argparse.ArgumentParser(description='운동 피드백 시스템 데모')
    parser.add_argument('--user-video', type=str, required=True,
                       help='사용자 운동 영상 경로')
    parser.add_argument('--gt-video', type=str, default=None,
                       help='Ground truth 영상 경로 (선택)')
    parser.add_argument('--model', type=str, default='./models/model.pt',
                       help='BiLSTM 모델 경로')
    parser.add_argument('--encoder', type=str, default='./models/label_encoder.pkl',
                       help='Label encoder 경로')
    parser.add_argument('--output', type=str, default='./output',
                       help='출력 디렉토리')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API 키 (환경변수 OPENAI_API_KEY 사용 가능)')
    
    args = parser.parse_args()
    
    # API 키 설정
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다. --api-key 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
    
    # 파일 존재 확인
    if not Path(args.user_video).exists():
        raise FileNotFoundError(f"사용자 영상을 찾을 수 없습니다: {args.user_video}")
    
    if not Path(args.model).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {args.model}")
    
    if not Path(args.encoder).exists():
        raise FileNotFoundError(f"Label encoder를 찾을 수 없습니다: {args.encoder}")
    
    print("="*60)
    print("통합 운동 피드백 시스템")
    print("="*60)
    print(f"사용자 영상: {args.user_video}")
    print(f"GT 영상: {args.gt_video or '(자동 선택)'}")
    print(f"출력 디렉토리: {args.output}")
    print("="*60 + "\n")
    
    # 시스템 초기화
    system = ExerciseFeedbackSystem(
        model_path=args.model,
        label_encoder_path=args.encoder,
        openai_api_key=api_key,
        model_type="gemini"  # 또는 "gpt"
    )
    
    # 비디오 처리
    result = system.process_video(
        user_video_path=args.user_video,
        gt_video_path=args.gt_video,
        output_dir=args.output
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("분석 결과")
    print("="*60)
    print(f"✅ 운동 종류: {result['exercise_type']}")
    print(f"✅ 신뢰도: {result['confidence']:.2%}")
    print(f"✅ 피드백 영상: {result['feedback_video']}")
    print(f"✅ JSON 요약: {result['json_summary']}")
    print("\n" + "="*60)
    print("피드백")
    print("="*60)
    print(result['feedback_text'])
    print("="*60)


if __name__ == "__main__":
    main()
