# -*- coding: utf-8 -*-
"""
LLM 기반 피드백 생성 모듈
"""

from openai import OpenAI
import google.generativeai as genai
from typing import Dict, Optional
import os
import re


class FeedbackGenerator:
    """운동별 맞춤 피드백 생성"""

    def __init__(self, api_key: str, model_type: str = "gpt"):
        self.model_type = model_type
        if model_type.lower() == "gemini":
            self.gemini_api_key = api_key
            genai.configure(api_key=api_key)
        else:
            self.openai_client = OpenAI(api_key=api_key)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        self.prompts_cache = {}
        self._exercise_name_map = {}  # 운동명 -> 파일명 매핑

    def _get_prompt_filename(self, exercise_type: str) -> str:
        """
        운동명을 프롬프트 파일명으로 변환

        Args:
            exercise_type: 운동 종류 (예: "squat", "lunge", "high knees")
                          분류 모델이 반환하는 영문 소문자 형식

        Returns:
            프롬프트 파일명 (예: "squat_prompts.md", "lunge_prompts.md", "high_knees_prompts.md")
        """
        # 캐시된 매핑이 있으면 사용
        if exercise_type in self._exercise_name_map:
            return self._exercise_name_map[exercise_type]

        filename = exercise_type.lower().replace(' ', '_') + '_prompts.md'

        self._exercise_name_map[exercise_type] = filename

        return filename

    def _load_prompt_sections(self, md_file: str) -> Dict[str, str]:
        """
        MD 파일에서 섹션별로 프롬프트 텍스트를 로드

        Args:
            md_file: MD 파일 경로

        Returns:
            섹션명을 키로, 내용을 값으로 하는 딕셔너리
        """
        if md_file in self.prompts_cache:
            return self.prompts_cache[md_file]

        filepath = os.path.join(self.prompts_dir, md_file)
        if not os.path.exists(filepath):
            return {}

        sections = {}
        current_section = None
        current_content = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 섹션 헤더 감지 (## section_name)
                section_match = re.match(r'^##\s+(\w+)\s*$', line)
                if section_match:
                    # 이전 섹션 저장
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    # 새 섹션 시작
                    current_section = section_match.group(1)
                    current_content = []
                elif current_section and not line.startswith('#'):
                    # 섹션 내용 수집
                    current_content.append(line.rstrip())

        # 마지막 섹션 저장
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        self.prompts_cache[md_file] = sections
        return sections
    
    def _build_prompt(self, summary_data: Dict, exercise: str, orientation: str) -> str:
        """
        범용 프롬프트 빌더 - MD 파일에서 자동으로 프롬프트 생성

        Args:
            summary_data: 자세 분석 결과
            exercise: 운동 종류
            orientation: 촬영 방향 (front/side)

        Returns:
            생성된 프롬프트
        """
        summary = summary_data["summary"]
        avg_scores = summary_data["average_joint_scores"]

        # 운동 종류에 맞는 프롬프트 파일명 가져오기
        prompt_filename = self._get_prompt_filename(exercise)

        # MD 파일에서 프롬프트 섹션 로드
        sections = self._load_prompt_sections(prompt_filename)

        # 프롬프트 파일이 없으면 기본 프롬프트 사용
        if not sections:
            sections = self._load_prompt_sections('default_prompts.md')

        # orientation에 따라 적절한 설명 선택
        orientation_key = f"{orientation}_orientation"
        orientation_desc = sections.get(orientation_key, '')

        # 기본 프롬프트 구조 생성
        prompt_parts = [
            f"너는 운동 코치 AI야. 아래는 사용자의 {exercise} 동작 평가 결과야.",
            "JSON 데이터를 보고, 사용자가 이해할 수 있게 피드백을 해줘.",
            ""
        ]

        # 운동 설명 추가
        if sections.get('exercise_description'):
            prompt_parts.append(f"** {exercise}에 대한 설명")
            prompt_parts.append(sections.get('exercise_description', ''))
            prompt_parts.append("")

        # orientation 설명 추가
        if orientation_desc:
            prompt_parts.append(orientation_desc)
            prompt_parts.append("")

        # 관절 설명 추가
        if sections.get('joint_descriptions'):
            prompt_parts.append(sections.get('joint_descriptions', ''))
            prompt_parts.append("")

        # 관절 점수 해석 규칙 추가 (있는 경우)
        if sections.get('joint_score_interpretation'):
            prompt_parts.append(sections.get('joint_score_interpretation', ''))
            prompt_parts.append("")

        # 평가 기준 추가
        if sections.get('evaluation_criteria'):
            prompt_parts.append(sections.get('evaluation_criteria', ''))
            prompt_parts.append("")

        # 결과 요약
        prompt_parts.append("# 결과 요약:")
        prompt_parts.append(f"라벨 비율 = {summary}")
        prompt_parts.append(f"평균 관절 점수 = {avg_scores}")
        prompt_parts.append("")

        # 출력 지침 추가
        if sections.get('output_guidelines'):
            # {orientation} 같은 플레이스홀더 치환
            output_guidelines = sections.get('output_guidelines', '')
            output_guidelines = output_guidelines.replace('{orientation}', orientation)
            output_guidelines = output_guidelines.replace('{exercise}', exercise)
            prompt_parts.append(output_guidelines)

        return '\n'.join(prompt_parts)

    def generate_feedback(self,
                         exercise_type: str,
                         summary_data: Dict,
                         orientation: str = "front") -> str:
        """
        운동 종류와 분석 데이터로 피드백 생성

        Args:
            exercise_type: 운동 종류 (예: "런지", "사이드 런지", "하이니즈" 등)
            summary_data: 자세 분석 결과 (JSON)
            orientation: 촬영 방향 (front/side)
            model_type: 사용할 LLM 모델 ("gpt" 또는 "gemini", 기본값: "gpt")

        Returns:
            피드백 텍스트
        """
        # 범용 프롬프트 빌더 사용
        prompt = self._build_prompt(summary_data, exercise_type, orientation)

        if self.model_type.lower() == "gemini":
            # Gemini 사용
            if not self.gemini_api_key:
                raise ValueError("Gemini API key가 설정되지 않았습니다. FeedbackGenerator 초기화 시 gemini_api_key를 제공해주세요.")

            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            return response.text
        else:
            # GPT 사용 (기본값)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
