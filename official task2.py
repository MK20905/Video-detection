import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.models.video import r3d_18
import subprocess
import numpy as np
import soundfile as sf
import io
import cv2
import matplotlib.pyplot as plt
import librosa
import torchvision.transforms as T
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
import warnings
import argparse
import torch.cuda.amp as amp

warnings.filterwarnings('ignore')

# -------------------------
# Loss Functions
# -------------------------

class BoundaryMatchingLoss(nn.Module):
    """Loss function for boundary detection"""
    def __init__(self, weight=1.0):
        super(BoundaryMatchingLoss, self).__init__()
        self.weight = weight
        
    def forward(self, predictions, targets):
        # Detect boundaries (transitions from 0->1 or 1->0)
        pred_boundaries = self._detect_boundaries(predictions)
        true_boundaries = self._detect_boundaries(targets.float())
        
        # L2 loss on boundary positions
        boundary_loss = F.mse_loss(pred_boundaries, true_boundaries)
        return self.weight * boundary_loss
    
    def _detect_boundaries(self, sequence):
        # sequence: (B, T)
        if sequence.dim() == 3:  # (B, T, 2) -> take argmax
            sequence = sequence.argmax(dim=-1).float()
        
        # Compute differences to find transitions
        padded = F.pad(sequence, (1, 0), value=0)
        boundaries = torch.abs(sequence - padded[:, :-1])
        return boundaries

class TemporalFocalLoss(nn.Module):
    """Focal loss adapted for temporal localization"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(TemporalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, T, 2), targets: (B, T)
        inputs_flat = inputs.reshape(-1, 2)  # Use reshape instead of view
        targets_flat = targets.view(-1)      # targets should be contiguous
        
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.view_as(targets)

# -------------------------
#  Proper AP@IoU Implementation
# -------------------------

def compute_ap_at_iou_levels(predictions, ground_truths, iou_thresholds=[0.5, 0.75, 0.9, 0.95]):
    """
    Compute Average Precision at different IoU thresholds
    """
    ap_results = {}
    
    for iou_thresh in iou_thresholds:
        all_scores = []
        all_labels = []
        
        for video_id in ground_truths:
            gt_segments = ground_truths[video_id]
            pred_segments = predictions.get(video_id, [])
            
            if not gt_segments:  # No ground truth segments
                continue
                
            # Sort predictions by confidence (descending)
            pred_segments = sorted(pred_segments, key=lambda x: x[0], reverse=True)
            
            # For each prediction, check if it matches any ground truth
            gt_matched = [False] * len(gt_segments)
            
            for pred_conf, pred_start, pred_end in pred_segments:
                all_scores.append(pred_conf)
                
                # Find best matching ground truth segment
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, (gt_start, gt_end) in enumerate(gt_segments):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if match meets IoU threshold
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    all_labels.append(1)  # True positive
                    gt_matched[best_gt_idx] = True
                else:
                    all_labels.append(0)  # False positive
        
        # Compute Average Precision
        if len(all_scores) > 0:
            ap = average_precision_score(all_labels, all_scores)
        else:
            ap = 0.0
            
        ap_results[f"AP@{iou_thresh}"] = ap
    
    return ap_results

def calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Calculate Intersection over Union for temporal segments"""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_ar_at_n(predictions, ground_truths, n_values=[50, 30, 20, 10, 5], iou_threshold=0.5):
    """
    Compute Average Recall at different numbers of proposals
    """
    ar_results = {n: [] for n in n_values}

    for video_id in ground_truths:
        gt_segments = ground_truths[video_id]
        pred_segments = predictions.get(video_id, [])
        
        if not gt_segments:  # Skip videos with no ground truth
            continue
            
        # Sort predictions by confidence (descending)
        pred_segments = sorted(pred_segments, key=lambda x: x[0], reverse=True)

        for n in n_values:
            top_n_preds = pred_segments[:n]
            matched_gt = set()
            
            for pred_conf, pred_start, pred_end in top_n_preds:
                for gt_idx, (gt_start, gt_end) in enumerate(gt_segments):
                    if gt_idx in matched_gt:
                        continue
                    if calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end) >= iou_threshold:
                        matched_gt.add(gt_idx)
                        break
            
            recall = len(matched_gt) / len(gt_segments) if gt_segments else 0
            ar_results[n].append(recall)

    # Average across all videos
    avg_ar = {f"AR@{n}": np.mean(scores) if scores else 0 for n, scores in ar_results.items()}
    return avg_ar

def compute_final_score(ap_results, ar_results):
    """
    Compute final competition score: (1/8)∑AP@IoU + (1/10)∑AR@N
    """
    ap_sum = sum([ap_results[f"AP@{iou}"] for iou in [0.5, 0.75, 0.9, 0.95]])
    ar_sum = sum([ar_results[f"AR@{n}"] for n in [50, 30, 20, 10, 5]])
    
    final_score = (1/8) * ap_sum + (1/10) * ar_sum
    return final_score

# -------------------------
# Improved Temporal Model
# -------------------------

class ImprovedTemporalDeepfakeLocalizer(nn.Module):
    """Improved model for temporal localization of deepfakes"""
    def __init__(self, use_audio=False, num_classes=2, max_frames=64):
        super().__init__()
        self.use_audio = use_audio
        self.num_classes = num_classes
        self.max_frames = max_frames
        
        # FIXED: Better video encoder with proper temporal features
        self.video_encoder = r3d_18(pretrained=True)
        # Remove the final classification layer
        self.video_encoder.fc = nn.Identity()
        
        # FIXED: Extract intermediate features for better temporal modeling
        self.video_feature_extractor = nn.Sequential(
            self.video_encoder.stem,
            self.video_encoder.layer1,
            self.video_encoder.layer2,
            self.video_encoder.layer3,
            self.video_encoder.layer4,
        )
        
        # Adaptive pooling to get consistent temporal dimension
        self.temporal_pool = nn.AdaptiveAvgPool3d((max_frames, 1, 1))
        
        # Audio encoders (if used)
        if self.use_audio:
            self.mfcc_encoder = AudioEncoder(13, 128, 64)
            self.mel_spec_encoder = AudioEncoder(80, 160, 80)
            self.waveform_encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=16),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.AdaptiveAvgPool1d(32),
                nn.Flatten(),
                nn.Linear(64 * 32, 64),
                nn.ReLU()
            )
            audio_feature_size = 64 + 80 + 64  # 208
        else:
            audio_feature_size = 0
        
        # Feature dimensions
        video_feature_size = 512  # R3D-18 feature size
        total_feature_size = video_feature_size + audio_feature_size
        
        # IMPROVED: Multi-scale temporal processing
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(total_feature_size, 256, kernel_size=3, padding=1),
            nn.Conv1d(total_feature_size, 256, kernel_size=5, padding=2),
            nn.Conv1d(total_feature_size, 256, kernel_size=7, padding=3),
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(256 * 3, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        # IMPROVED: Better temporal transformer
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # IMPROVED: Boundary-aware detection head
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )
        
        # Enhanced confidence scoring
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def encode_audio(self, audio_features):
        """Encode audio features"""
        encodings = []
        encodings.append(self.mfcc_encoder(audio_features['mfcc']))
        encodings.append(self.mel_spec_encoder(audio_features['mel_spec']))
        
        waveform = audio_features['waveform'].unsqueeze(1)
        encodings.append(self.waveform_encoder(waveform))
        
        return torch.cat(encodings, dim=1)
    
    def forward(self, video, audio_features=None):
        batch_size = video.size(0)
        
        # FIXED: Proper video feature extraction with temporal preservation
        # video: (B, C, T, H, W)
        video_features = self.video_feature_extractor(video)  # (B, 512, T', H', W')
        
        # Pool spatial dimensions, keep temporal
        video_features = self.temporal_pool(video_features)  # (B, 512, max_frames, 1, 1)
        video_features = video_features.squeeze(-1).squeeze(-1)  # (B, 512, max_frames)
        video_features = video_features.permute(0, 2, 1)  # (B, max_frames, 512)
        
        # Add audio features if available
        if self.use_audio and audio_features is not None:
            audio_encoding = self.encode_audio(audio_features)  # (B, audio_dim)
            audio_temporal = audio_encoding.unsqueeze(1).expand(-1, self.max_frames, -1)
            temporal_features = torch.cat([video_features, audio_temporal], dim=2)
        else:
            temporal_features = video_features
        
        # Multi-scale processing
        conv_input = temporal_features.permute(0, 2, 1)  # (B, features, T)
        multi_scale_outputs = []
        for conv_layer in self.multi_scale_conv:
            multi_scale_outputs.append(conv_layer(conv_input))
        
        # Fuse multi-scale features
        fused_features = torch.cat(multi_scale_outputs, dim=1)  # (B, 768, T)
        fused_features = self.feature_fusion(fused_features)  # (B, 512, T)
        
        # Apply temporal transformer
        transformer_input = fused_features.permute(0, 2, 1)  # (B, T, 512)
        enhanced_features = self.temporal_transformer(transformer_input)  # (B, T, 512)
        
        # Boundary detection
        conv_input = enhanced_features.permute(0, 2, 1)  # (B, 512, T)
        boundary_logits = self.boundary_detector(conv_input)  # (B, num_classes, T)
        boundary_logits = boundary_logits.permute(0, 2, 1)  # (B, T, num_classes)
        
        # Confidence scoring
        confidence_scores = self.confidence_head(enhanced_features)  # (B, T, 1)
        confidence_scores = confidence_scores.squeeze(-1)  # (B, T)
        
        return boundary_logits, confidence_scores

# -------------------------
# Better Dataset Implementation
# -------------------------

class ImprovedTemporalLocalizationDataset(Dataset):
    def __init__(self, video_dir, metadata_file, max_frames=64, fps=25, 
                 audio_duration=5.0, sample_rate=16000, use_audio=False, mode='train'):
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.fps = fps
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.use_audio = use_audio
        self.mode = mode
        
        # FIXED: Better metadata loading
        if isinstance(metadata_file, str) and metadata_file.endswith('.json'):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        elif isinstance(metadata_file, dict):
            self.metadata = self._create_metadata_from_dict(metadata_file)
        else:
            raise ValueError("metadata_file must be a JSON file path or dictionary")
        
        # Data transforms
        if mode == 'train':
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.3),
                T.RandomRotation(5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _create_metadata_from_dict(self, label_dict):
        """Create metadata from simple label dict - IMPROVED with realistic segments"""
        metadata = []
        for vid_id, label in label_dict.items():
            # For demo purposes, create more realistic fake segments
            if label == 1:  # Fake video
                # Generate random but realistic fake segments
                np.random.seed(int(vid_id))  # Consistent segments for same video
                num_segments = np.random.randint(1, 4)  # 1-3 fake segments
                fake_segments = []
                
                for _ in range(num_segments):
                    start = np.random.uniform(0.5, 8.0)  # Random start time
                    duration = np.random.uniform(1.0, 3.0)  # Random duration
                    fake_segments.append([start, start + duration])
            else:
                fake_segments = []
            
            metadata.append({
                'file': f"{vid_id}.mp4",
                'modify_type': 'visual_modified' if label == 1 else 'real',
                'fake_segments': fake_segments,
                'video_frames': self.max_frames * 2,
                'label': label
            })
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def _read_video_with_temporal_info(self, video_path, fake_segments):
        """IMPROVED: Better video reading with proper temporal alignment"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return self._get_dummy_video_and_labels()
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            
            if total_frames < 1:
                cap.release()
                return self._get_dummy_video_and_labels()
            
            # Better frame sampling strategy
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
                # Pad the indices if necessary
                while len(frame_indices) < self.max_frames:
                    frame_indices.append(frame_indices[-1] if frame_indices else 0)
            else:
                # Uniform sampling across the video
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            
            frames = []
            frame_labels = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
                ret, frame = cap.read()
                
                if not ret:
                    # Use last valid frame if read fails
                    if frames:
                        frame = frames[-1]
                    else:
                        frame = np.zeros((112, 112, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (112, 112))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = self.transform(Image.fromarray(frame))
                frames.append(frame)
                
                # Determine label based on temporal segments
                frame_time = idx / video_fps
                is_fake = self._is_frame_fake(frame_time, fake_segments)
                frame_labels.append(1 if is_fake else 0)
            
            cap.release()
            
            # Ensure exact length
            frames = frames[:self.max_frames]
            frame_labels = frame_labels[:self.max_frames]
            
            video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
            labels_tensor = torch.tensor(frame_labels, dtype=torch.long)
            
            return video_tensor, labels_tensor
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return self._get_dummy_video_and_labels()
    
    def _get_dummy_video_and_labels(self):
        """Generate dummy video and labels when processing fails"""
        dummy_frames = torch.zeros(3, self.max_frames, 112, 112)
        dummy_labels = torch.zeros(self.max_frames, dtype=torch.long)
        return dummy_frames, dummy_labels
    
    def _is_frame_fake(self, frame_time, fake_segments):
        """Check if a frame at given time is within fake segments"""
        for start_time, end_time in fake_segments:
            if start_time <= frame_time <= end_time:
                return True
        return False
    
    def _extract_audio_features(self, video_path):
        """Extract audio features (simplified version)"""
        if not self.use_audio:
            return self._get_dummy_audio_features()
        
        try:
            command = [
                'ffmpeg', '-i', video_path, '-f', 'wav', '-ac', '1',
                '-ar', str(self.sample_rate), '-loglevel', 'quiet', 'pipe:1'
            ]
            proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            audio_buffer = io.BytesIO(proc.stdout)
            y, sr = sf.read(audio_buffer)

            if len(y) == 0:
                return self._get_dummy_audio_features()
            
            # Process audio
            max_length = int(self.sample_rate * self.audio_duration)
            if len(y) > max_length:
                y = y[:max_length]
            elif len(y) < max_length:
                y = np.pad(y, (0, max_length - len(y)))

            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)

            features = {
                'mfcc': torch.tensor(mfcc, dtype=torch.float32),
                'mel_spec': torch.tensor(mel_spec, dtype=torch.float32),
                'waveform': torch.tensor(y, dtype=torch.float32)
            }
            return features

        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return self._get_dummy_audio_features()
    
    def _get_dummy_audio_features(self):
        """Generate dummy audio features"""
        target_length = int(self.audio_duration * self.sample_rate)
        time_frames = target_length // 512 + 1
        
        return {
            'mfcc': torch.zeros(13, time_frames),
            'mel_spec': torch.zeros(80, time_frames),
            'waveform': torch.zeros(target_length)
        }
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        video_file = item['file']
        fake_segments = item.get('fake_segments', [])
        
        video_path = os.path.join(self.video_dir, video_file)
        
        # Get video and frame-level labels
        video_tensor, frame_labels = self._read_video_with_temporal_info(video_path, fake_segments)
        
        # Get audio features if requested
        audio_features = None
        if self.use_audio:
            audio_features = self._extract_audio_features(video_path)
        
        if self.use_audio:
            return video_tensor, audio_features, frame_labels, fake_segments
        else:
            return video_tensor, frame_labels, fake_segments

# -------------------------
# Custom Collate Function
# -------------------------

def collate_fn(batch):
    """Custom collate function to handle variable-length video tensors"""
    if len(batch[0]) == 4:  # With audio
        videos, audio_features, frame_labels, fake_segments = zip(*batch)
        use_audio = True
    else:  # Without audio
        videos, frame_labels, fake_segments = zip(*batch)
        use_audio = False

    # Find the maximum length in the batch
    max_len = max(v.size(1) for v in videos)  # Get max temporal dimension (T)

    # Pad videos and frame_labels to max_len
    padded_videos = torch.stack([F.pad(v, (0, 0, 0, 0, 0, max_len - v.size(1))) for v in videos])
    padded_labels = torch.stack([F.pad(l, (0, max_len - l.size(0))) for l in frame_labels])

    if use_audio:
        # Handle audio features (assume they are already fixed size or pad if needed)
        # For simplicity, assume audio features are pre-processed to a fixed size in _extract_audio_features
        audio_batch = {}
        for key in audio_features[0].keys():
            audio_batch[key] = torch.stack([af[key] for af in audio_features])
        return padded_videos, audio_batch, padded_labels, list(fake_segments)
    else:
        return padded_videos, padded_labels, list(fake_segments)

# -------------------------
# Better Evaluation
# -------------------------

def evaluate_temporal_model_improved(model, dataloader, device):
    """IMPROVED evaluation with proper metrics"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    video_predictions = {}
    video_ground_truths = {}

    focal_loss = TemporalFocalLoss(alpha=1, gamma=2)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if model.use_audio:
                videos, audio_features, frame_labels, fake_segments = batch_data
                for key in audio_features:
                    audio_features[key] = audio_features[key].to(device)
            else:
                videos, frame_labels, fake_segments = batch_data
                audio_features = None
            
            videos = videos.to(device)
            frame_labels = frame_labels.to(device)
            
            boundary_logits, confidence_scores = model(videos, audio_features)
            loss = focal_loss(boundary_logits, frame_labels)
            total_loss += loss.item()
            
            probs = F.softmax(boundary_logits, dim=-1)[:, :, 1]
            
            for i in range(videos.size(0)):
                video_idx = batch_idx * videos.size(0) + i
                video_id = f"{video_idx:06d}.mp4"
                
                # Extract segments from predictions
                segments = extract_segments_from_predictions(
                    probs[i].cpu().numpy(), 
                    confidence_scores[i].cpu().numpy(), 
                    threshold=0.3,  # Lower threshold for better recall
                    min_duration=0.5
                )
                
                video_predictions[video_id] = segments
                video_ground_truths[video_id] = fake_segments[i] if isinstance(fake_segments[i], list) else fake_segments
                
                # Collect frame-level predictions for overall AP
                all_predictions.extend(probs[i].cpu().numpy())
                all_labels.extend(frame_labels[i].cpu().numpy())

    # Compute metrics
    frame_ap = average_precision_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else 0.0
    ap_iou_results = compute_ap_at_iou_levels(video_predictions, video_ground_truths)
    ar_results = compute_ar_at_n(video_predictions, video_ground_truths)
    final_score = compute_final_score(ap_iou_results, ar_results)
    
    return {
        'loss': total_loss / len(dataloader),
        'frame_ap': frame_ap,
        'final_score': final_score,
        **ap_iou_results,
        **ar_results
    }

# -------------------------
# AudioEncoder class 
# -------------------------

class AudioEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Conv1d(hidden_size//2, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def extract_segments_from_predictions(fake_probs, confidences, threshold=0.5, min_duration=0.5, fps=25):
    """Extract fake segments from frame-level predictions"""
    segments = []
    
    # Find contiguous segments above threshold
    above_threshold = fake_probs > threshold
    
    start_idx = None
    for i, is_fake in enumerate(above_threshold):
        if is_fake and start_idx is None:
            start_idx = i
        elif not is_fake and start_idx is not None:
            # End of segment
            duration = (i - start_idx) / fps
            if duration >= min_duration:
                avg_confidence = confidences[start_idx:i].mean()
                start_time = start_idx / fps
                end_time = i / fps
                segments.append([float(avg_confidence), float(start_time), float(end_time)])
            start_idx = None
    
    # Handle case where video ends while in a fake segment
    if start_idx is not None:
        duration = (len(fake_probs) - start_idx) / fps
        if duration >= min_duration:
            avg_confidence = confidences[start_idx:].mean()
            start_time = start_idx / fps
            end_time = len(fake_probs) / fps
            segments.append([float(avg_confidence), float(start_time), float(end_time)])
    
    # Sort by confidence (descending)
    segments.sort(key=lambda x: x[0], reverse=True)
    
    return segments

# -------------------------
# Training Function
# -------------------------

def train_temporal_model_improved(model, train_loader, val_loader=None, num_epochs=15, device='cuda',
                                learning_rate=1e-4, log_dir='runs/temporal_improved', patience=7):
    
    # Improved loss combination
    focal_loss = TemporalFocalLoss(alpha=2, gamma=2.5)  # More aggressive focal loss
    boundary_loss = BoundaryMatchingLoss(weight=0.3)
    
    # Better optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)
    scaler = amp.GradScaler()
    
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_score = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_focal_loss = 0.0
        train_boundary_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            if model.use_audio:
                videos, audio_features, frame_labels, _ = batch_data
                for key in audio_features:
                    audio_features[key] = audio_features[key].to(device)
            else:
                videos, frame_labels, _ = batch_data
                audio_features = None
            
            videos = videos.to(device)
            frame_labels = frame_labels.to(device)
            
            optimizer.zero_grad()
            
            with amp.autocast():
                boundary_logits, confidence_scores = model(videos, audio_features)
                
                # Improved loss calculation
                focal_loss_val = focal_loss(boundary_logits, frame_labels)
                boundary_loss_val = boundary_loss(boundary_logits, frame_labels)
                
                # Confidence regularization
                confidence_reg = 0.01 * torch.mean((confidence_scores - 0.5) ** 2)
                
                # Total loss with regularization
                total_loss = focal_loss_val + boundary_loss_val + confidence_reg
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + batch_idx / len(train_loader))
            
            train_loss += total_loss.item()
            train_focal_loss += focal_loss_val.item()
            train_boundary_loss += boundary_loss_val.item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Focal': f'{focal_loss_val.item():.4f}',
                'Boundary': f'{boundary_loss_val.item():.4f}'
            })
        
        # Average losses
        train_loss /= len(train_loader)
        train_focal_loss /= len(train_loader)
        train_boundary_loss /= len(train_loader)
        
        # Logging
        writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Train_Focal', train_focal_loss, epoch)
        writer.add_scalar('Loss/Train_Boundary', train_boundary_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_metrics = evaluate_temporal_model_improved(model, val_loader, device)
            
            # Log all metrics
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            print(f"          Val Final Score: {val_metrics['final_score']:.4f}")
            print(f"          Val Frame AP: {val_metrics['frame_ap']:.4f}")
            print(f"          Val AP@0.5: {val_metrics['AP@0.5']:.4f}")
            print(f"          Val AR@50: {val_metrics['AR@50']:.4f}")
            
            # Use final competition score for best model selection
            current_score = val_metrics['final_score']
            
            if current_score > best_val_score:
                best_val_score = current_score
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
                print(f"          🎯 New best score: {best_val_score:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    writer.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation score: {best_val_score:.4f}")
    
    return model

# -------------------------
# Prediction Generation
# -------------------------

def generate_task2_predictions_improved(model, test_loader, device, output_file, video_filenames=None):
    """Generate Task 2 format predictions with proper video naming"""
    model.eval()
    predictions = {}
    
    print("Generating Task 2 predictions...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Processing videos")):
            if model.use_audio:
                videos, audio_features, _, fake_segments = batch_data
                for key in audio_features:
                    audio_features[key] = audio_features[key].to(device)
            else:
                videos, _, fake_segments = batch_data
                audio_features = None
            
            videos = videos.to(device)
            boundary_logits, confidence_scores = model(videos, audio_features)
            
            # Get predictions
            probs = F.softmax(boundary_logits, dim=-1)
            fake_probs = probs[:, :, 1]  # (B, T)
            confidences = confidence_scores  # (B, T)
            
            for i in range(videos.size(0)):
                video_idx = batch_idx * videos.size(0) + i
                
                # Use provided filenames or generate default ones
                if video_filenames and video_idx < len(video_filenames):
                    video_id = video_filenames[video_idx]
                else:
                    video_id = f"{video_idx:06d}.mp4"
                
                # Extract segments with multiple thresholds for better coverage
                segments = []
                for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    thresh_segments = extract_segments_from_predictions(
                        fake_probs[i].cpu().numpy(),
                        confidences[i].cpu().numpy(),
                        threshold=threshold,
                        min_duration=0.4
                    )
                    segments.extend(thresh_segments)
                
                # Remove duplicates and sort by confidence
                unique_segments = []
                for seg in segments:
                    is_duplicate = False
                    for existing in unique_segments:
                        # Check for significant overlap
                        overlap = calculate_temporal_iou(seg[1], seg[2], existing[1], existing[2])
                        if overlap > 0.7:
                            is_duplicate = True
                            # Keep the one with higher confidence
                            if seg[0] > existing[0]:
                                unique_segments.remove(existing)
                                unique_segments.append(seg)
                            break
                    if not is_duplicate:
                        unique_segments.append(seg)
                
                # Sort by confidence and limit number of segments
                unique_segments.sort(key=lambda x: x[0], reverse=True)
                unique_segments = unique_segments[:10]  # Limit to top 10 segments
                
                predictions[video_id] = unique_segments
    
    # Save predictions in Task 2 format
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Task 2 predictions saved to {output_file}")
    
    # Print statistics
    total_videos = len(predictions)
    videos_with_segments = sum(1 for segments in predictions.values() if segments)
    total_segments = sum(len(segments) for segments in predictions.values())
    
    print(f"📊 Prediction Statistics:")
    print(f"   Total videos: {total_videos}")
    print(f"   Videos with detected segments: {videos_with_segments}")
    print(f"   Total segments detected: {total_segments}")
    print(f"   Average segments per video: {total_segments/total_videos:.2f}")

# -------------------------
# Main Execution with Better Error Handling
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Improved Temporal Deepfake Localizer")
    parser.add_argument('--video-dir', default="/Users/mk/Desktop/CASTLE2024/day1_downloads",
                        help="Directory containing video files")
    parser.add_argument('--metadata-file', default=None,
                        help="Path to metadata JSON file")
    parser.add_argument('--use-audio', action='store_true', default=False,
                        help="Use audio features in the model")
    parser.add_argument('--num-epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=2, help="Batch size")
    parser.add_argument('--max-frames', type=int, default=64, help="Maximum frames per video")
    parser.add_argument('--output-file', default="task2_predictions_improved.json",
                        help="File to save predictions")
    parser.add_argument('--save-model', default="temporal_localizer_improved.pt",
                        help="Path to save trained model")
    
    args = parser.parse_args()

    # Default label dict for testing
    label_dict = {
        "08": 1, "09": 0, "10": 1, "11": 1,
        "12": 1, "13": 0, "14": 1, "15": 0,
        "16": 0, "17": 0, "18": 0, "19": 0, "20": 0
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    try:
        print("📂 Creating improved temporal localization dataset...")
        dataset = ImprovedTemporalLocalizationDataset(
            args.video_dir, 
            args.metadata_file or label_dict, 
            max_frames=args.max_frames, 
            use_audio=args.use_audio, 
            mode='train'
        )
        
        print(f"✅ Dataset loaded with {len(dataset)} videos")
        
        # Better data splitting
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
        
        print(f"📊 Data split: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        # Create improved model
        print("🏗️ Creating improved temporal localization model...")
        model = ImprovedTemporalDeepfakeLocalizer(
            use_audio=args.use_audio, 
            max_frames=args.max_frames
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Train the model
        print("🎯 Starting improved temporal localization training...")
        trained_model = train_temporal_model_improved(
            model, train_loader, val_loader, 
            num_epochs=args.num_epochs, device=device, 
            learning_rate=args.learning_rate
        )
        
        # Final evaluation
        print("\n📊 Final evaluation on validation set...")
        final_metrics = evaluate_temporal_model_improved(trained_model, val_loader, device)
        
        print("🎯 Final Competition Metrics:")
        print(f"   Final Score: {final_metrics['final_score']:.4f}")
        print(f"   Frame AP: {final_metrics['frame_ap']:.4f}")
        
        print("📈 AP@IoU Metrics:")
        for iou in [0.5, 0.75, 0.9, 0.95]:
            print(f"   AP@{iou}: {final_metrics[f'AP@{iou}']:.4f}")
        
        print("🎯 AR@N Metrics:")
        for n in [50, 30, 20, 10, 5]:
            print(f"   AR@{n}: {final_metrics[f'AR@{n}']:.4f}")
        
        # Save the model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {
                'use_audio': args.use_audio,
                'num_classes': 2,
                'max_frames': args.max_frames
            },
            'final_metrics': final_metrics,
            'training_args': vars(args)
        }, args.save_model)
        
        print(f"💾 Model saved to {args.save_model}")
        
        # Generate predictions
        print("\n🔮 Generating Task 2 predictions...")
        generate_task2_predictions_improved(
            trained_model, val_loader, device, args.output_file
        )
        
        print("✅ Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()