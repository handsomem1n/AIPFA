# -*- coding: utf-8 -*-
"""
BiLSTM 기반 운동 분류 모듈
지원하는 운동: Squat, Lunge, Side Lunge, Situp, High Knees, Bridge, Cobra, Jumping Jack
"""

import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple


class PoseBiLSTMClassifier(nn.Module):
    """BiLSTM 기반 자세 분류 모델"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=2, 
            bidirectional=True,
            dropout=0.3, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat((h_fwd, h_bwd), dim=1)
        return self.fc(h)


class ExerciseClassifier:
    """운동 종류 분류기"""
    
    def __init__(self, model_path: str, label_encoder_path: str):
        """
        Args:
            model_path: 학습된 BiLSTM 모델 경로
            label_encoder_path: Label encoder 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Label encoder 로딩
        self.le = joblib.load(label_encoder_path)
        
        # 모델 로딩
        self.model = PoseBiLSTMClassifier(
            input_size=132,  # 33 joints × (x, y, z, v)
            hidden_size=64,
            num_classes=len(self.le.classes_)
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, pose_data: pd.DataFrame) -> Tuple[str, float]:
        """
        자세 데이터로부터 운동 종류 예측
        
        Args:
            pose_data: 자세 데이터 (columns: x0~x32, y0~y32, z0~z32, v0~v32)
            
        Returns:
            (운동 종류, 신뢰도)
        """
        # 데이터 전처리
        pose_seq = pose_data.loc[:, 'x0':'v32'].values
        pose_tensor = torch.tensor(pose_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        length_tensor = torch.tensor([len(pose_seq)]).to(self.device)
        
        # 예측
        with torch.no_grad():
            output = self.model(pose_tensor, length_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
            
            pred_label = self.le.classes_[pred_idx.item()]
            confidence_score = confidence.item()
        
        return pred_label, confidence_score
    
    def predict_batch(self, pose_sequences: list) -> list:
        """
        여러 시퀀스에 대해 일괄 예측
        
        Args:
            pose_sequences: 자세 시퀀스 리스트
            
        Returns:
            [(운동 종류, 신뢰도), ...]
        """
        results = []
        for seq in pose_sequences:
            pred_label, confidence = self.predict(seq)
            results.append((pred_label, confidence))
        return results
