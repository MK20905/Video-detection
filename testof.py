"""
Unified Deepfake Detection Script
Supports:
- Task 1: Video-level Deepfake Detection (AUC)
- Task 2: Temporal Localization (AP and AR)

Run with:
    python combined_task1_task2.py --task 1   # for video-level
    python combined_task1_task2.py --task 2   # for temporal localization
    --video-dir <path> --metadata-file <path> --use-audio --num-epochs 10 --learning-rate 1e-4
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
import warnings
import argparse
import torch.cuda.amp as amp

warnings.filterwarnings('ignore')

# -------------------------
# Utility Functions
# -------------------------

def read_video_frames(video_path, max_frames, start_frame, transform):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    filename = os.path.basename(video_path)
    if filename.startswith("08"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10500)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    available_frames = int(total_frames - current_pos)
    
    if available_frames < max_frames:
        frame_indices = range(available_frames)
    else:
        frame_indices = np.linspace(0, available_frames-1, max_frames, dtype=int)
    
    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + target_idx)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frame = frames[-1]
            else:
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(Image.fromarray(frame))
        frames.append(frame)
    
    cap.release()
    
    while len(frames) < max_frames:
        if frames:
            frames.append(torch.zeros_like(frames[0]))
        else:
            frames.append(torch.zeros(3, 112, 112))
    
    return torch.stack(frames).permute(1, 0, 2, 3)

def extract_audio_features(video_path, audio_duration, sample_rate):
    try:
        command = ['ffmpeg', '-i', video_path, '-f', 'wav', '-ac', '1', '-ar', str(sample_rate), '-loglevel', 'quiet', 'pipe:1']
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_buffer = io.BytesIO(proc.stdout)
        y, sr = sf.read(audio_buffer)
        if len(y) == 0:
            return get_dummy_audio_features(audio_duration, sample_rate)
        max_length = int(sample_rate * audio_duration)
        if len(y) > max_length:
            y = y[:max_length]
        elif len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        return {
            'mfcc': torch.tensor(mfcc, dtype=torch.float32),
            'mel_spec': torch.tensor(mel_spec, dtype=torch.float32),
            'waveform': torch.tensor(y, dtype=torch.float32)
        }
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return get_dummy_audio_features(audio_duration, sample_rate)

def get_dummy_audio_features(audio_duration, sample_rate):
    target_length = int(sample_rate * audio_duration)
    time_frames = target_length // 512 + 1
    return {
        'mfcc': torch.zeros(13, time_frames),
        'mel_spec': torch.zeros(80, time_frames),
        'waveform': torch.zeros(target_length)
    }

# -------------------------
# Loss Functions
# -------------------------

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (pred.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class BoundaryMatchingLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions, targets):
        pred_boundaries = self._detect_boundaries(predictions)
        true_boundaries = self._detect_boundaries(targets.float())
        boundary_loss = F.mse_loss(pred_boundaries, true_boundaries)
        return self.weight * boundary_loss
    
    def _detect_boundaries(self, sequence):
        if sequence.dim() == 3:
            sequence = sequence.argmax(dim=-1).float()
        padded = F.pad(sequence, (1, 0), value=0)
        boundaries = torch.abs(sequence - padded[:, :-1])
        return boundaries

class TemporalFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs_flat = inputs.reshape(-1, 2)
        targets_flat = targets.view(-1)
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# -------------------------
# Metric Functions
# -------------------------

def compute_auc(predictions, labels):
    probs = F.softmax(predictions, dim=-1)[:, :, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

def compute_ap_at_iou_levels(predictions, ground_truths, iou_thresholds=[0.5, 0.75, 0.9, 0.95]):
    ap_results = {}
    for iou_thresh in iou_thresholds:
        all_scores = []
        all_labels = []
        for video_id in ground_truths:
            gt_segments = ground_truths[video_id]
            pred_segments = predictions.get(video_id, [])
            if not gt_segments:
                continue
            pred_segments = sorted(pred_segments, key=lambda x: x[0], reverse=True)
            gt_matched = [False] * len(gt_segments)
            for pred_conf, pred_start, pred_end in pred_segments:
                all_scores.append(pred_conf)
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, (gt_start, gt_end) in enumerate(gt_segments):
                    if gt_matched[gt_idx]:
                        continue
                    iou = calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                all_labels.append(1 if best_iou >= iou_thresh and best_gt_idx >= 0 else 0)
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
        ap = average_precision_score(all_labels, all_scores) if all_scores else 0.0
        ap_results[f"AP@{iou_thresh}"] = ap
    return ap_results

def calculate_temporal_iou(pred_start, pred_end, gt_start, gt_end):
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    if intersection_start >= intersection_end:
        return 0.0
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0.0

def compute_ar_at_n(predictions, ground_truths, n_values=[50, 30, 20, 10, 5], iou_threshold=0.5):
    ar_results = {n: [] for n in n_values}
    for video_id in ground_truths:
        gt_segments = ground_truths[video_id]
        pred_segments = predictions.get(video_id, [])
        if not gt_segments:
            continue
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
            ar_results[n].append(len(matched_gt) / len(gt_segments) if gt_segments else 0)
    return {f"AR@{n}": np.mean(scores) if scores else 0 for n, scores in ar_results.items()}

def compute_final_score(ap_results, ar_results):
    ap_sum = sum(ap_results[f"AP@{iou}"] for iou in [0.5, 0.75, 0.9, 0.95])
    ar_sum = sum(ar_results[f"AR@{n}"] for n in [50, 30, 20, 10, 5])
    return (1/8) * ap_sum + (1/10) * ar_sum

# -------------------------
# Dataset
# -------------------------

class DeepfakeDataset(Dataset):
    def __init__(self, video_dir, metadata_file, max_frames=32, start_frame=200, 
                 audio_duration=5.0, sample_rate=16000, use_audio=True, task='task1', mode='train'):
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.use_audio = use_audio
        self.task = task
        self.mode = mode
        
        if isinstance(metadata_file, str) and metadata_file.endswith('.json'):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        elif isinstance(metadata_file, dict):
            self.metadata = self._create_metadata_from_dict(metadata_file)
        else:
            raise ValueError("metadata_file must be a JSON file path or dictionary")
        
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.3 if mode == 'train' else 0),
            T.RandomRotation(5 if mode == 'train' else 0),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if mode == 'train' else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_metadata_from_dict(self, label_dict):
        metadata = []
        for vid_id, label in label_dict.items():
            if self.task == 'task2' and label == 1:
                np.random.seed(int(vid_id))
                num_segments = np.random.randint(1, 4)
                fake_segments = [[np.random.uniform(0.5, 8.0), np.random.uniform(1.0, 3.0) + s] for s in np.random.uniform(0.5, 8.0, num_segments)]
            else:
                fake_segments = []
            metadata.append({
                'file': f"{vid_id}.mp4",
                'label': label,
                'fake_segments': fake_segments
            })
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def _is_frame_fake(self, frame_time, fake_segments, fps):
        for start_time, end_time in fake_segments:
            if start_time <= frame_time <= end_time:
                return True
        return False
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        video_path = os.path.join(self.video_dir, item['file'])
        frames = read_video_frames(video_path, self.max_frames, self.start_frame, self.transform)
        audio_features = extract_audio_features(video_path, self.audio_duration, self.sample_rate) if self.use_audio else None
        if self.task == 'task2':
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()
            frame_labels = torch.tensor([1 if self._is_frame_fake(i / fps, item['fake_segments'], fps) else 0 for i in range(self.max_frames)], dtype=torch.long)
            return frames, audio_features, frame_labels, item['fake_segments']
        else:
            label = torch.tensor(item['label'], dtype=torch.float32)
            return frames, audio_features, label

def collate_fn(batch):
    if len(batch[0]) == 4:
        videos, audio_features, frame_labels, fake_segments = zip(*batch)
        use_audio = True
    else:
        videos, frame_labels, fake_segments = zip(*batch)
        use_audio = False
    max_len = max(v.size(1) for v in videos)
    padded_videos = torch.stack([F.pad(v, (0, 0, 0, 0, 0, max_len - v.size(1))) for v in videos])
    if use_audio:
        audio_batch = {key: torch.stack([af[key] for af in audio_features]) for key in audio_features[0].keys()}
        return padded_videos, audio_batch, frame_labels, list(fake_segments)
    return padded_videos, frame_labels, list(fake_segments)

# -------------------------
# Models
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

class DeepfakeDetector(nn.Module):
    def __init__(self, use_audio=True, task='task1', max_frames=64):
        super().__init__()
        self.use_audio = use_audio
        self.task = task
        self.max_frames = max_frames
        
        self.video_encoder = r3d_18(pretrained=True)
        self.video_encoder.fc = nn.Identity()
        
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
            audio_feature_size = 64 + 80 + 64
        else:
            audio_feature_size = 0
        
        video_feature_size = 512
        total_feature_size = video_feature_size + audio_feature_size
        
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(total_feature_size, 256, kernel_size=3, padding=1),
            nn.Conv1d(total_feature_size, 256, kernel_size=5, padding=2),
            nn.Conv1d(total_feature_size, 256, kernel_size=7, padding=3),
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(256 * 3, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True),
            num_layers=4
        )
        
        self.boundary_detector = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 2, kernel_size=1)
        )
        
        self.video_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
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
        encodings = [self.mfcc_encoder(audio_features['mfcc']), self.mel_spec_encoder(audio_features['mel_spec'])]
        waveform = audio_features['waveform'].unsqueeze(1)
        encodings.append(self.waveform_encoder(waveform))
        return torch.cat(encodings, dim=1)
    
    def forward(self, video, audio_features=None):
        video_features = self.video_encoder(video)
        video_features = video_features.mean([-1, -2])
        
        if self.use_audio and audio_features is not None:
            audio_encoding = self.encode_audio(audio_features)
            audio_temporal = audio_encoding.unsqueeze(1).expand(-1, self.max_frames, -1)
            temporal_features = torch.cat([video_features, audio_temporal], dim=2)
        else:
            temporal_features = video_features
        
        conv_input = temporal_features.permute(0, 2, 1)
        multi_scale_outputs = [conv_layer(conv_input) for conv_layer in self.multi_scale_conv]
        fused_features = torch.cat(multi_scale_outputs, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        transformer_input = fused_features.permute(0, 2, 1)
        enhanced_features = self.temporal_transformer(transformer_input)
        
        if self.task == 'task1':
            confidence = self.video_classifier(enhanced_features.mean(dim=1))
            return confidence.squeeze(-1)
        else:
            boundary_logits = self.boundary_detector(enhanced_features.permute(0, 2, 1)).permute(0, 2, 1)
            confidence_scores = self.confidence_head(enhanced_features).squeeze(-1)
            return boundary_logits, confidence_scores

# -------------------------
# Training Functions
# -------------------------

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate, log_dir, task):
    criterion = LabelSmoothingLoss() if task == 'task2' else nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir)
    
    best_metric = 0.0
    best_model_state = None
    history = {'train_losses': [], 'train_metrics': [], 'val_losses': [], 'val_metrics': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_metric = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_data in pbar:
            if model.use_audio:
                videos, audio_features, labels, _ = batch_data
                for key in audio_features:
                    audio_features[key] = audio_features[key].to(device)
            else:
                videos, labels, _ = batch_data
                audio_features = None
            
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with amp.autocast():
                if task == 'task2':
                    outputs = model(videos, audio_features)[0]
                    if outputs.shape[1] != labels.shape[1]:
                        outputs = F.interpolate(outputs.permute(0, 2, 1), size=labels.size(1), mode='linear').permute(0, 2, 1)
                    loss = criterion(outputs.reshape(-1, 2), labels.reshape(-1))
                else:
                    outputs = model(videos, audio_features)
                    loss = criterion(outputs, labels)
                metric = (outputs > 0.5).eq(labels).float().mean() if task == 'task1' else 0
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_metric += metric.item() * labels.numel() if task == 'task1' else 0
            total_samples += labels.numel()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc' if task == 'task1' else 'Metric': f'{metric.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        avg_train_metric = total_metric / total_samples if task == 'task1' else 0
        history['train_losses'].append(avg_train_loss)
        history['train_metrics'].append(avg_train_metric)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Metric/train', avg_train_metric, epoch)
        
        if val_loader is not None:
            val_loss, val_metric = evaluate_model(model, val_loader, device, task)
            history['val_losses'].append(val_loss)
            history['val_metrics'].append(val_metric)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metric/val', val_metric, epoch)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_metric:.4f}, Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            if (task == 'task1' and val_metric > best_metric) or (task == 'task2' and val_metric['final_score'] > best_metric):
                best_metric = val_metric if task == 'task1' else val_metric['final_score']
                best_model_state = model.state_dict().copy()
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_metric:.4f}")
    
    writer.close()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, history

def evaluate_model(model, dataloader, device, task):
    model.eval()
    total_loss = 0
    total_metric = 0
    criterion = LabelSmoothingLoss() if task == 'task2' else nn.BCELoss()
    
    with torch.no_grad():
        for batch_data in dataloader:
            if model.use_audio:
                videos, audio_features, labels, _ = batch_data
                for key in audio_features:
                    audio_features[key] = audio_features[key].to(device)
            else:
                videos, labels, _ = batch_data
                audio_features = None
            
            videos, labels = videos.to(device), labels.to(device)
            
            if task == 'task2':
                outputs = model(videos, audio_features)[0]
                if outputs.shape[1] != labels.shape[1]:
                    outputs = F.interpolate(outputs.permute(0, 2, 1), size=labels.size(1), mode='linear').permute(0, 2, 1)
                loss = criterion(outputs.reshape(-1, 2), labels.reshape(-1))
                metric = compute_auc(outputs, labels)
            else:
                outputs = model(videos, audio_features)
                loss = criterion(outputs, labels)
                metric = (outputs > 0.5).eq(labels).float().mean().item()
            total_loss += loss.item()
            total_metric += metric
    
    return total_loss / len(dataloader), total_metric / len(dataloader)

# -------------------------
# Prediction Generation
# -------------------------

def generate_predictions(model, test_loader, device, output_file, task, video_filenames=None):
    model.eval()
    if task == 'task1':
        predictions = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Generating Task 1 predictions")):
                if model.use_audio:
                    videos, audio_features, _ = batch_data
                    for key in audio_features:
                        audio_features[key] = audio_features[key].to(device)
                else:
                    videos, _, _ = batch_data
                    audio_features = None
                videos = videos.to(device)
                confidence = model(videos, audio_features)
                for i in range(videos.size(0)):
                    video_idx = batch_idx * videos.size(0) + i
                    video_id = video_filenames[video_idx] if video_filenames and video_idx < len(video_filenames) else f"{video_idx:06d}.mp4"
                    predictions.append(f"{video_id};{confidence[i].item():.4f}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(predictions))
        print(f"✅ Task 1 predictions saved to {output_file}")
    else:
        predictions = {}
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Generating Task 2 predictions")):
                if model.use_audio:
                    videos, audio_features, _, fake_segments = batch_data
                    for key in audio_features:
                        audio_features[key] = audio_features[key].to(device)
                else:
                    videos, _, fake_segments = batch_data
                    audio_features = None
                videos = videos.to(device)
                boundary_logits, confidence_scores = model(videos, audio_features)
                probs = F.softmax(boundary_logits, dim=-1)[:, :, 1]
                for i in range(videos.size(0)):
                    video_idx = batch_idx * videos.size(0) + i
                    video_id = video_filenames[video_idx] if video_filenames and video_idx < len(video_filenames) else f"{video_idx:06d}.mp4"
                    segments = []
                    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                        thresh_segments = [(conf.mean(), start / 25, end / 25) for conf, start, end in 
                                         [(confidence_scores[i][start_idx:end_idx].mean().item(), start_idx, end_idx) 
                                          for start_idx in range(len(probs[i])) 
                                          if probs[i][start_idx] > threshold 
                                          for end_idx in range(start_idx + 1, len(probs[i]) + 1) 
                                          if all(probs[i][j] > threshold for j in range(start_idx, end_idx)) 
                                          and (end_idx - start_idx) / 25 >= 0.4]]
                        segments.extend(thresh_segments)
                    unique_segments = sorted([(c, s, e) for c, s, e in segments if not any(calculate_temporal_iou(s, e, es, ee) > 0.7 for cs, es, ee in unique_segments) or c > next((cs for cs, es, ee in unique_segments if calculate_temporal_iou(s, e, es, ee) > 0.7), 0)], key=lambda x: x[0], reverse=True)[:10]
                    predictions[video_id] = [[c, s, e] for c, s, e in unique_segments]
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✅ Task 2 predictions saved to {output_file}")
        total_videos = len(predictions)
        videos_with_segments = sum(1 for segs in predictions.values() if segs)
        total_segments = sum(len(segs) for segs in predictions.values())
        print(f"📊 Prediction Statistics: Total videos: {total_videos}, Videos with segments: {videos_with_segments}, Total segments: {total_segments}, Avg segments per video: {total_segments/total_videos:.2f}")

# -------------------------
# Visualization Functions
# -------------------------

def plot_training_history(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    if history['val_losses']:
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['train_metrics'], 'b-', label='Training Metric')
    if history['val_metrics']:
        ax2.plot(epochs, history['val_metrics'], 'r-', label='Validation Metric')
    ax2.set_title('Training and Validation Metric')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metric')
    ax2.legend()
    ax2.grid(True)
    ax3.text(0.5, 0.5, 'Additional metrics can be plotted here', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Additional Metrics')
    ax4.text(0.5, 0.5, 'Model info:\n• Video: R3D-18\n• Audio: Multi-feature CNN\n• Fusion: Multi-scale', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Model Architecture')
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataset, idx, device, task):
    model.eval()
    if model.use_audio:
        if task == 'task2':
            video, audio_features, labels, _ = dataset[idx]
        else:
            video, audio_features, labels = dataset[idx]
        for key in audio_features:
            audio_features[key] = audio_features[key].unsqueeze(0).to(device)
    else:
        if task == 'task2':
            video, labels, _ = dataset[idx]
        else:
            video, labels = dataset[idx]
        audio_features = None
    
    with torch.no_grad():
        if task == 'task2':
            outputs = model(video.unsqueeze(0).to(device), audio_features)[0]
            if outputs.shape[1] != labels.shape[1]:
                outputs = F.interpolate(outputs.permute(0, 2, 1), size=labels.size(1), mode='linear').permute(0, 2, 1)
            pred = outputs.squeeze(0).argmax(dim=1).cpu()
        else:
            outputs = model(video.unsqueeze(0).to(device), audio_features)
            pred = (outputs.squeeze(0).item() > 0.5).int()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    if task == 'task2':
        ax1.plot(labels.numpy(), label='Ground Truth', color='green', linewidth=2)
        ax1.plot(pred.numpy(), label='Prediction', color='red', linestyle='--', linewidth=2)
        ax1.set_title(f'Frame-level Prediction for Video {idx} (Task 2)')
    else:
        ax1.text(0.5, 0.5, f'Video-level Prediction: {pred.item()}\nTrue Label: {labels.item()}', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Video-level Prediction for Video {idx} (Task 1)')
    ax1.set_xlabel('Frame Index' if task == 'task2' else 'N/A')
    ax1.set_ylabel('Fake=1 / Real=0')
    ax1.legend()
    ax1.grid(True)
    frame = video[:, 0, :, :].permute(1, 2, 0).cpu().numpy()
    frame = frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    frame = np.clip(frame, 0, 1)
    ax2.imshow(frame)
    ax2.set_title('Sample Frame (First frame)')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------
# Main Execution
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Deepfake Detection")
    parser.add_argument('--task', type=int, choices=[1, 2], default=2, help="Task number (1 or 2)")
    parser.add_argument('--video-dir', default="/Users/mk/Desktop/CASTLE2024/day1_downloads", help="Directory containing video files")
    parser.add_argument('--metadata-file', default=None, help="Path to metadata JSON file")
    parser.add_argument('--use-audio', action='store_true', default=False, help="Use audio features")
    parser.add_argument('--num-epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=2, help="Batch size")
    parser.add_argument('--max-frames', type=int, default=64, help="Maximum frames per video")
    parser.add_argument('--output-file', default="predictions.txt" if task == 1 else "task2_predictions.json", help="File to save predictions")
    parser.add_argument('--save-model', default="deepfake_detector.pt", help="Path to save trained model")
    
    args = parser.parse_args()
    
    label_dict = {
        "08": 1, "09": 0, "10": 1, "11": 1,
        "12": 1, "13": 0, "14": 1, "15": 0,
        "16": 0, "17": 0, "18": 0, "19": 0, "20": 0
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    try:
        print("📂 Creating dataset...")
        dataset = DeepfakeDataset(
            args.video_dir, args.metadata_file or label_dict,
            max_frames=args.max_frames, use_audio=args.use_audio, task=f'task{args.task}', mode='train'
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
        
        print(f"📊 Data split: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        print("🏗️ Creating model...")
        model = DeepfakeDetector(use_audio=args.use_audio, task=f'task{args.task}', max_frames=args.max_frames).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        print(f"🎯 Starting task{args.task} training...")
        trained_model, history = train_model(model, train_loader, val_loader, args.num_epochs, device, args.learning_rate, 'runs/experiment', f'task{args.task}')
        
        print("\n📊 Final evaluation...")
        val_loss, val_metric = evaluate_model(trained_model, val_loader, device, f'task{args.task}')
        print(f"🎯 Final Metric: {val_metric:.4f}" if args.task == 1 else f"🎯 Final Score: {val_metric['final_score']:.4f}")
        
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {'use_audio': args.use_audio, 'task': f'task{args.task}', 'max_frames': args.max_frames},
            'final_metric': val_metric,
            'training_args': vars(args)
        }, args.save_model)
        print(f"💾 Model saved to {args.save_model}")
        
        print("\n🔮 Generating predictions...")
        generate_predictions(trained_model, val_loader, device, args.output_file, f'task{args.task}')
        
        print("\n🔍 Visualizing results...")
        visualize_predictions(trained_model, dataset, 0, device, f'task{args.task}')
        
        print("✅ Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()