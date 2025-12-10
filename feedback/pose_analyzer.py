# -*- coding: utf-8 -*-
"""
Mediapipe 기반 자세 분석 모듈
"""

import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.cm as cm
import json
from typing import List, Dict, Tuple


POSE_LANDMARKS = mp.solutions.pose.PoseLandmark

# 주요 관절쌍
VEC_PAIRS = [
    (POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW),
    (POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST),
    (POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW),
    (POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST),
    (POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE),
    (POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE),
    (POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE),
    (POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE),
    (POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER),
    (POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP),
]

# 시각화용 스켈레톤 선분
EDGES = [
    (int(POSE_LANDMARKS.LEFT_SHOULDER), int(POSE_LANDMARKS.RIGHT_SHOULDER)),
    (int(POSE_LANDMARKS.LEFT_SHOULDER), int(POSE_LANDMARKS.LEFT_ELBOW)),
    (int(POSE_LANDMARKS.LEFT_ELBOW), int(POSE_LANDMARKS.LEFT_WRIST)),
    (int(POSE_LANDMARKS.RIGHT_SHOULDER), int(POSE_LANDMARKS.RIGHT_ELBOW)),
    (int(POSE_LANDMARKS.RIGHT_ELBOW), int(POSE_LANDMARKS.RIGHT_WRIST)),
    (int(POSE_LANDMARKS.LEFT_HIP), int(POSE_LANDMARKS.RIGHT_HIP)),
    (int(POSE_LANDMARKS.LEFT_HIP), int(POSE_LANDMARKS.LEFT_KNEE)),
    (int(POSE_LANDMARKS.LEFT_KNEE), int(POSE_LANDMARKS.LEFT_ANKLE)),
    (int(POSE_LANDMARKS.RIGHT_HIP), int(POSE_LANDMARKS.RIGHT_KNEE)),
    (int(POSE_LANDMARKS.RIGHT_KNEE), int(POSE_LANDMARKS.RIGHT_ANKLE)),
]


class PoseAnalyzer:
    """자세 분석 및 피드백 생성"""
    
    def __init__(self):
        self.cmap = cm.get_cmap('turbo')
    
    def extract_landmarks(self, video_path: str, max_frames=99999, stride=1) -> pd.DataFrame:
        """
        비디오에서 관절 데이터 추출
        
        Returns:
            DataFrame with columns: frame_id, x0~x32, y0~y32, z0~z32, v0~v32
        """
        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(video_path)
        
        data_list = []
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or idx >= max_frames:
                    break
                
                if idx % stride != 0:
                    idx += 1
                    continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                
                if res.pose_landmarks:
                    h, w = frame.shape[:2]
                    
                    # 관절 데이터 추출
                    row_data = {'frame_id': idx}
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        row_data[f'x{i}'] = lm.x * w
                        row_data[f'y{i}'] = lm.y * h
                        row_data[f'z{i}'] = lm.z
                        row_data[f'v{i}'] = lm.visibility
                    
                    data_list.append(row_data)
                
                idx += 1
        
        cap.release()
        return pd.DataFrame(data_list)
    
    def _extract_landmarks_for_eval(self, video_path: str, max_frames=99999, stride=1) -> List[Dict]:
        """평가용 관절 추출 (프레임 포함)"""
        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or idx >= max_frames:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                
                if res.pose_landmarks:
                    h, w = frame.shape[:2]
                    pts = []
                    for lm in res.pose_landmarks.landmark:
                        pts.append((lm.x*w, lm.y*h, lm.visibility))
                    frames.append({"pts": np.array(pts, dtype=np.float32), "frame": frame})
                
                idx += 1
        
        cap.release()
        return frames
    
    def _unit_vec(self, v):
        n = np.linalg.norm(v) + 1e-8
        return v / n
    
    def _vectorize(self, pts):
        """관절을 벡터로 변환"""
        vecs = []
        for a, b in VEC_PAIRS:
            pa = np.array(pts[int(a)][:2])
            pb = np.array(pts[int(b)][:2])
            vecs.append(self._unit_vec(pb - pa))
        return np.concatenate(vecs, axis=0)
    
    def _bbox_scale(self, pts):
        """바운딩 박스 스케일"""
        xs, ys = pts[:, 0], pts[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        return math.sqrt(w*w + h*h) + 1e-6
    
    def _oks_score(self, ptsA, ptsB, k=0.5):
        """OKS 점수"""
        s = self._bbox_scale(ptsA)
        d2 = np.sum((ptsA[:, :2] - ptsB[:, :2])**2, axis=1)
        oks = np.exp(-d2 / (2 * (s**2) * (k**2)))
        return float(np.mean(oks))
    
    def _cosine01(self, a, b):
        """코사인 유사도 [0, 1]"""
        c = float(np.dot(a, b) / ((np.linalg.norm(a)+1e-8) * (np.linalg.norm(b)+1e-8)))
        return 0.5 * (c + 1.0)
    
    def _l2_distance(self, a_pts, b_pts):
        """L2 거리"""
        scale = self._bbox_scale(a_pts)
        return float(np.linalg.norm((a_pts[:,:2] - b_pts[:,:2]).reshape(-1)) / scale)
    
    def _best_match_idx(self, gt_vecs, usr_vecs, center, win=5):
        """GT 프레임 주변에서 최적 매칭 찾기"""
        lo = max(0, center - win)
        hi = min(len(usr_vecs) - 1, center + win)
        best_i, best_s = center, -1.0
        
        for j in range(lo, hi + 1):
            s = self._cosine01(gt_vecs[center], usr_vecs[j])
            if s > best_s:
                best_s, best_i = s, j
        
        return best_i, best_s
    
    def _classify_frame(self, gt_i, best_j, l2_val, l2_th=0.5, speed_tol=2):
        """프레임 분류 (NG/Fast/Slow/Good)"""
        if l2_val > l2_th:
            return "NG"
        delta = best_j - gt_i
        if delta > speed_tol:
            return "Fast"
        if delta < -speed_tol:
            return "Slow"
        return "Good"
    
    def _compute_score_range(self, all_scores, low_p=5, high_p=95):
        """점수 범위 계산"""
        # 빈 배열 처리
        if len(all_scores) == 0:
            return 0.0, 1.0

        smin = np.percentile(all_scores, low_p)
        smax = np.percentile(all_scores, high_p)
        if smax <= smin:
            smin, smax = 0.0, 1.0
        return smin, smax
    
    def _score_to_color(self, score: float, smin: float, smax: float):
        """점수를 색상으로 변환"""
        x = (score - smin) / (smax - smin + 1e-8)
        x = float(np.clip(x, 0.0, 1.0))
        rgba = self.cmap(x)
        r, g, b, a = [int(val * 255) for val in rgba]
        return (b, g, r)  # OpenCV BGR
    
    def _pairwise_cosine(self, pa1, pb1, pa2, pb2):
        """부위별 코사인 유사도"""
        v1 = self._unit_vec(pa1 - pb1)
        v2 = self._unit_vec(pa2 - pb2)
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return 0.5 * (c + 1.0)
    
    def _draw_skeleton_per_edge(self, img, pts_gt, pts_usr, smin=0.0, smax=1.0):
        """부위별 색상으로 스켈레톤 그리기"""
        for a, b in EDGES:
            pa_gt, pb_gt = pts_gt[a][:2], pts_gt[b][:2]
            pa_usr, pb_usr = pts_usr[a][:2], pts_usr[b][:2]
            
            score = self._pairwise_cosine(
                np.array(pa_gt), np.array(pb_gt),
                np.array(pa_usr), np.array(pb_usr)
            )
            color = self._score_to_color(score, smin, smax)
            
            cv2.line(img, 
                    (int(pts_usr[a][0]), int(pts_usr[a][1])),
                    (int(pts_usr[b][0]), int(pts_usr[b][1])), 
                    color, 3)
            cv2.circle(img, (int(pts_usr[a][0]), int(pts_usr[a][1])), 4, color, -1)
            cv2.circle(img, (int(pts_usr[b][0]), int(pts_usr[b][1])), 4, color, -1)
    
    def evaluate(self, 
                 gt_video: str, 
                 usr_video: str,
                 win=5, 
                 speed_tol=2, 
                 l2_th=0.5,
                 save_path="out_feedback.mp4",
                 full_json_path="out_feedback_full.json",
                 summary_json_path="out_feedback_summary.json"):
        """
        GT와 사용자 영상 비교 평가
        """
        # 관절 추출
        gt_frames = self._extract_landmarks_for_eval(gt_video)
        usr_frames = self._extract_landmarks_for_eval(usr_video)
        gt_vecs = [self._vectorize(f["pts"]) for f in gt_frames]
        usr_vecs = [self._vectorize(f["pts"]) for f in usr_frames]
        
        # 1차 패스: 점수 분포 수집
        all_scores = []
        N = min(len(gt_frames), len(usr_frames))
        for i in range(N):
            j, cos_s = self._best_match_idx(gt_vecs, usr_vecs, i, win=win)
            oks = self._oks_score(gt_frames[i]["pts"], usr_frames[j]["pts"])
            final_score = cos_s * oks
            all_scores.append(final_score)
        
        smin, smax = self._compute_score_range(all_scores)
        
        # 2차 패스: 시각화 + 통계
        H, W = gt_frames[0]["frame"].shape[:2]
        fps = int(cv2.VideoCapture(gt_video).get(cv2.CAP_PROP_FPS)) or 20
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W*2, H))
        
        stats = {"Good": 0, "Fast": 0, "Slow": 0, "NG": 0}
        frame_results = []
        joint_accumulator = {}
        
        for i in range(N):
            j, cos_s = self._best_match_idx(gt_vecs, usr_vecs, i, win=win)
            oks = self._oks_score(gt_frames[i]["pts"], usr_frames[j]["pts"])
            final_score = cos_s * oks
            l2v = self._l2_distance(gt_frames[i]["pts"], usr_frames[j]["pts"])
            label = self._classify_frame(i, j, l2v, l2_th=l2_th, speed_tol=speed_tol)
            stats[label] += 1
            
            # 부위별 점수
            joint_scores = {}
            for a, b in EDGES:
                pa_gt, pb_gt = gt_frames[i]["pts"][a][:2], gt_frames[i]["pts"][b][:2]
                pa_usr, pb_usr = usr_frames[j]["pts"][a][:2], usr_frames[j]["pts"][b][:2]
                score = self._pairwise_cosine(
                    np.array(pa_gt), np.array(pb_gt),
                    np.array(pa_usr), np.array(pb_usr)
                )
                joint_scores[f"{a}-{b}"] = round(float(score), 3)
                joint_accumulator.setdefault(f"{a}-{b}", []).append(score)
            
            frame_results.append({
                "frame_idx": i,
                "match_idx": j,
                "label": label,
                "final_score": round(float(final_score), 3),
                "joint_scores": joint_scores
            })
            
            # 시각화
            left = gt_frames[i]["frame"].copy()
            right = usr_frames[j]["frame"].copy()
            
            self._draw_skeleton_per_edge(left, gt_frames[i]["pts"], gt_frames[i]["pts"], smin, smax)
            self._draw_skeleton_per_edge(right, gt_frames[i]["pts"], usr_frames[j]["pts"], smin, smax)
            
            cv2.putText(left, f"GT idx={i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(right, f"USR idx={j}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.putText(right, f"{label} score={final_score:.2f}", (10,70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            right_resized = cv2.resize(right, (W, H))
            out.write(np.hstack([left, right_resized]))
        
        out.release()
        
        # JSON 저장
        total = sum(stats.values())
        summary_data = {
            "summary": {k: round(v/total*100, 1) for k, v in stats.items()},
            "average_joint_scores": {k: round(float(np.mean(v)), 3) for k, v in joint_accumulator.items()}
        }
        
        with open(full_json_path, 'w', encoding='utf-8') as f:
            json.dump(frame_results, f, ensure_ascii=False, indent=2)
        
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"  -> 영상 저장: {save_path}")
        print(f"  -> JSON 저장: {full_json_path}, {summary_json_path}")
