import cv2
import time
import numpy as np
import requests
import sys
import os
import socket
import argparse
import signal
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
import pickle
import json
import shutil
import uuid
from datetime import datetime, date
from flask import Flask, jsonify, request, Response, send_from_directory
import threading
from werkzeug.utils import secure_filename
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

try:
    import torchreid
    from torchreid.utils import FeatureExtractor
    REID_AVAILABLE = True
except Exception:
    torchreid = None
    FeatureExtractor = None
    REID_AVAILABLE = False

data_lock = threading.Lock()
shutdown_event = threading.Event()

def request_shutdown(signum=None, frame=None):
    shutdown_event.set()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("\nShutdown requested. Stopping monitoring...")

try:
    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)
except Exception:
    pass

# ==================== Person Re-Identification (ReID) ====================
class ReIDManager:
    def __init__(self, threshold=0.75):
        self.enabled = REID_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = None
        if self.enabled:
            # Using OSNet (osnet_x1_0) - specialized for person re-identification
            # FeatureExtractor handles model loading and preprocessing (resize/norm)
            self.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                device=self.device,
                verbose=False
            )
        
        self.threshold = threshold
        # Identity Store: persistent_id -> {'embedding': tensor, 'last_seen': timestamp}
        self.identity_bank = {}
        self.next_persistent_id = 1
        self.bank_file = os.path.join(BASE_DIR, "reid_bank.pickle")
        self.load_bank()

    def load_bank(self):
        if os.path.exists(self.bank_file):
            try:
                with open(self.bank_file, "rb") as f:
                    data = pickle.load(f)
                    
                if isinstance(data, dict):
                    if 'bank' in data:
                        self.identity_bank = data['bank']
                        self.next_persistent_id = data.get('next_id', 1)
                    else:
                        # Legacy format: the whole pickle was the bank
                        self.identity_bank = data
                        # Estimate next_id from keys like "Person_N"
                        pids = [int(k.split('_')[1]) for k in data.keys() if isinstance(k, str) and k.startswith("Person_")]
                        self.next_persistent_id = max(pids) + 1 if pids else 1
                
                # Validation: Remove invalid entries that would cause KeyError
                valid_bank = {}
                for pid, entry in self.identity_bank.items():
                    if isinstance(entry, dict) and 'embedding' in entry:
                        valid_bank[pid] = entry
                    elif isinstance(entry, np.ndarray):
                        # Very old format where entry WAS the embedding
                        valid_bank[pid] = {'embedding': entry, 'last_seen': time.time()}
                
                self.identity_bank = valid_bank
                print(f"Loaded {len(self.identity_bank)} valid identities from ReID bank.")
            except Exception as e:
                print(f"Error loading ReID bank: {e}")

    def save_bank(self):
        try:
            with open(self.bank_file, "wb") as f:
                pickle.dump({'bank': self.identity_bank, 'next_id': self.next_persistent_id}, f)
        except Exception as e:
            print(f"Error saving ReID bank: {e}")

    @torch.no_grad()
    def get_embedding(self, person_crop):
        if not self.enabled or self.extractor is None:
            return None
        if person_crop is None or person_crop.size == 0: return None
        # Convert BGR (OpenCV) to RGB (expected by torchreid/PIL)
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        # extractor returns a torch tensor
        features = self.extractor([rgb_crop])
        # L2 Normalize for cosine similarity matching
        features = nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()

    def match_identity(self, current_embedding, current_sig=None):
        if current_embedding is None: return None
        
        best_id = None
        max_sim = -1
        
        for pid, data in self.identity_bank.items():
            # 1. Try ReID Embedding Match
            person_sim = -1
            if 'embedding' in data:
                # Moving Average format
                gallery_emb = data['embedding']
                if gallery_emb.shape == current_embedding.shape:
                    person_sim = np.dot(current_embedding, gallery_emb)
            elif 'embeddings' in data and data['embeddings']:
                # Legacy Gallery format: Find best match
                for gallery_emb in data['embeddings']:
                    if gallery_emb.shape == current_embedding.shape:
                        sim = np.dot(current_embedding, gallery_emb)
                        if sim > person_sim: person_sim = sim
            
            if person_sim > max_sim:
                max_sim = person_sim
                best_id = pid
        
        # Threshold check for ReID
        if max_sim > self.threshold:
            # Update Identity Bank with Moving Average (Stability Patch)
            # Formula: 0.8 * old + 0.2 * new, then re-normalize
            if 'embedding' not in self.identity_bank[best_id]:
                # Convert legacy to moving average format
                self.identity_bank[best_id]['embedding'] = self.identity_bank[best_id]['embeddings'][0]
            
            old_emb = self.identity_bank[best_id]['embedding']
            updated_emb = 0.8 * old_emb + 0.2 * current_embedding
            # L2 Normalize the updated embedding
            norm = np.linalg.norm(updated_emb)
            if norm > 0:
                self.identity_bank[best_id]['embedding'] = updated_emb / norm
            
            self.identity_bank[best_id]['last_seen'] = time.time()
            return best_id
        
        # 2. Fallback to Clothing Color (Stability Patch)
        if current_sig is not None:
            for pid, data in self.identity_bank.items():
                if 'color_sig' in data:
                    if compare_signatures(current_sig, data['color_sig']) > 0.6:
                        # Found a match via color! Update its last seen
                        self.identity_bank[pid]['last_seen'] = time.time()
                        return pid

        # 3. New identity
        new_id = f"Person_{self.next_persistent_id}"
        self.next_persistent_id += 1
        self.identity_bank[new_id] = {
            'embedding': current_embedding,
            'color_sig': current_sig,
            'last_seen': time.time(),
            'first_seen': time.time()
        }
        return new_id

    def add_to_gallery(self, pid, embedding):
        """Add a new angle to a person's signature if it's sufficiently different"""
        if pid not in self.identity_bank or embedding is None: return
        
        gallery = self.identity_bank[pid].setdefault('embeddings', [])
        
        # Only add if it's a 'new' angle (sim < 0.90 compared to existing ones)
        # and gallery is not too large (max 10 angles)
        is_new_angle = True
        for gallery_emb in gallery:
            if np.dot(embedding, gallery_emb) > 0.90:
                is_new_angle = False
                break
        
        if is_new_angle and len(gallery) < 10:
            gallery.append(embedding)
            print(f"📸 Captured new body angle for {pid} (Total: {len(gallery)})")

    def prune_bank(self, max_idle=86400, min_duration=5, protected_ids=None):
        """Remove short-lived 'ghost' IDs to prevent bank bloat. Default idle increased to 24h."""
        now = time.time()
        to_delete = []
        protected_ids = protected_ids or []
        for pid, data in self.identity_bank.items():
            if pid in protected_ids:
                continue # Never prune named individuals
                
            idle_time = now - data['last_seen']
            duration = data['last_seen'] - data.get('first_seen', data['last_seen'])
            
            # If seen once and never again for 24 hours, or tracked for < 5s then gone
            if idle_time > max_idle or (idle_time > 300 and duration < min_duration):
                to_delete.append(pid)
        
        for pid in to_delete:
            del self.identity_bank[pid]
        if to_delete:
            print(f"🧹 Pruned {len(to_delete)} ghost identities from ReID bank.")
        return to_delete

reid_manager = ReIDManager(threshold=0.45)
tracker_to_persistent = {} # Maps YOLO tracker_id -> persistent_id
IDENTITY_MODE = "reid" if REID_AVAILABLE else "tracker"
YOLO_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SINGLE_PERSON_MODE = True
SINGLE_PERSON_LABEL = "Resident"
SETTINGS_FILE = os.path.join(BASE_DIR, "system_settings.json")

DEFAULT_SETTINGS = {
    "bot_token": "",
    "chat_id": "",
    "message_cooldown_sec": 90,
    "fall_confirm_window_sec": 10.0,
    "preferred_camera": "0",
    "max_people_to_track": 4,
    "enable_telegram": False,
    "enable_detection": True,
    "enable_voice_alert": False,
    "enable_wellness_monitoring": True,
    "display_metrics_overlay": True,
    "deployment_mode": "server",
    "node_id": socket.gethostname(),
    "central_server_url": "http://127.0.0.1:5000",
    "server_bind_host": "0.0.0.0",
    "server_port": 5000
}

MAJOR_FALL_TELEGRAM_BURST_COUNT = 3
MAJOR_FALL_TELEGRAM_BURST_DELAY_SEC = 0.35
LOW_POWER_IDLE_TIMEOUT_SEC = 3.0
LOW_POWER_PEEK_INTERVAL_SEC = 0.25
WAKE_GRACE_PERIOD_SEC = 2.5
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
STREAM_MAX_WIDTH = 640
STREAM_JPEG_QUALITY = 70
STREAM_MAX_FPS = 12.0
PREVIEW_MAX_FPS = 20.0
PREVIEW_WINDOW_NAME = "Elderly Monitor Live Overlay"
PREVIEW_WINDOW_WIDTH = 800
PREVIEW_WINDOW_HEIGHT = 450
LIVE_DETECTION_CONFIDENCE = 0.55
OFFLINE_DETECTION_CONFIDENCE = 0.65
OFFLINE_FALLBACK_DETECTION_CONFIDENCE = 0.35
LIVE_INFERENCE_IMGSZ = 512
OFFLINE_INFERENCE_IMGSZ = 960
OFFLINE_FALLBACK_INFERENCE_IMGSZ = 1280
MULTI_PERSON_MIN_TRACK_COUNT = 4
MOTION_REFERENCE_FPS = 30.0
LYING_HORIZONTAL_ANGLE_MIN = 68.0
LYING_TRANSITION_ANGLE_MIN = 50.0
LYING_HORIZONTAL_BOX_RATIO = 1.10
LYING_TRANSITION_BOX_RATIO = 1.00
LYING_MAX_MOTION_SCORE = 6.0
LYING_MAX_ANGLE_RATE = 22.0
LYING_STABLE_SECONDS = 0.9
LYING_TRANSITION_GRACE_SECONDS = 0.8
LYING_BODY_SPREAD_TRANSITION_RATIO = 1.05
LYING_BODY_SPREAD_STABLE_RATIO = 1.15
LYING_FROM_TRANSITION_SECONDS = 1.8
DOWN_STATE_HOLD_SECONDS = 2.5
SLEEPING_AFTER_LYING_SECONDS = 8.0
SUDDEN_DROP_VERTICAL_SPEED = 7.0
SUDDEN_DROP_MOTION_SCORE = 12.0
SUDDEN_DROP_HOLD_SECONDS = 1.2

settings = DEFAULT_SETTINGS.copy()
system_events = []
last_telegram_sent_at = {}
last_notified_activity = {}
multi_person_notice_sent = False
latest_stream_frame = None
preview_window_enabled = True
preview_window_initialized = False
last_stream_update_at = 0.0
last_preview_update_at = 0.0
telegram_dashboard_message_id = None
telegram_dashboard_last_text = ""
last_telegram_dashboard_update_at = 0.0
ward_profile = {}
ward_locked_persistent_id = None
monitored_persistent_id = None
pending_registration = {
    "active": False,
    "name": "",
    "preferred_target_id": None,
    "requested_at": 0.0
}
WARD_REGISTRATION_STEPS = [
    {"code": "front", "label": "Front View", "hint": "Face the camera directly."},
    {"code": "left", "label": "Left Side View", "hint": "Turn your left shoulder toward the camera."},
    {"code": "right", "label": "Right Side View", "hint": "Turn your right shoulder toward the camera."},
    {"code": "back", "label": "Back View", "hint": "Stand with your back facing the camera."}
]
WARD_GALLERY_DIR = os.path.join(BASE_DIR, "registered_people")
ward_registration_session = {
    "active": False,
    "target_id": None,
    "name": "",
    "captures": 0,
    "required_captures": 220,
    "expires_at": 0.0,
    "last_capture_at": 0.0,
    "started_at": 0.0,
    "get_ready_until": 0.0,
    "current_step": 0,
    "steps": WARD_REGISTRATION_STEPS,
    "captured_steps": [],
    "auto_mode": True,
    "capture_interval_sec": 0.08,
    "last_saved_index": 0,
    "gallery_dir": "",
    "completed_at": 0.0,
    "completed_name": "",
    "completed_captures": 0
}
latest_person_crops = {}
latest_person_pose_meta = {}
remote_nodes = {}
remote_edge_reports = {}
evaluation_metrics = {
    "fps": 0.0,
    "people_tracked": 0,
    "low_power_entries": 0,
    "wake_events": 0,
    "last_sleep_started_at": None,
    "last_wake_latency_sec": None,
    "avg_wake_latency_sec": None,
    "fall_alerts_total": 0,
    "minor_falls_total": 0,
    "major_falls_total": 0,
    "recoveries_total": 0,
    "central_sync_ok": 0,
    "central_sync_fail": 0
}
multi_person_mode_active = False
multi_person_count = 0

def add_system_event(message, level="info"):
    entry = {
        "message": str(message),
        "level": level,
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S", time.localtime())
    }
    with data_lock:
        system_events.append(entry)
        del system_events[:-100]
    print(f"[{level.upper()}] {message}")

def load_settings():
    global settings
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                settings.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
        except Exception as e:
            print(f"Error loading settings: {e}")
    settings["message_cooldown_sec"] = int(float(settings.get("message_cooldown_sec", 90) or 90))
    settings["fall_confirm_window_sec"] = float(settings.get("fall_confirm_window_sec", 10.0) or 10.0)
    settings["preferred_camera"] = str(settings.get("preferred_camera", "0") or "0")
    settings["max_people_to_track"] = min(10, max(1, int(float(settings.get("max_people_to_track", 4) or 4))))
    settings["deployment_mode"] = str(settings.get("deployment_mode", DEFAULT_SETTINGS["deployment_mode"]) or DEFAULT_SETTINGS["deployment_mode"]).strip().lower()
    if settings["deployment_mode"] not in {"server", "edge", "standalone"}:
        settings["deployment_mode"] = DEFAULT_SETTINGS["deployment_mode"]
    settings["node_id"] = str(settings.get("node_id", DEFAULT_SETTINGS["node_id"]) or DEFAULT_SETTINGS["node_id"]).strip()
    settings["central_server_url"] = str(settings.get("central_server_url", DEFAULT_SETTINGS["central_server_url"]) or DEFAULT_SETTINGS["central_server_url"]).strip().rstrip("/")
    settings["server_bind_host"] = str(settings.get("server_bind_host", DEFAULT_SETTINGS["server_bind_host"]) or DEFAULT_SETTINGS["server_bind_host"]).strip()
    settings["server_port"] = int(float(settings.get("server_port", DEFAULT_SETTINGS["server_port"]) or DEFAULT_SETTINGS["server_port"]))
    for key in ["enable_telegram", "enable_detection", "enable_voice_alert", "enable_wellness_monitoring", "display_metrics_overlay"]:
        settings[key] = bool(settings.get(key, DEFAULT_SETTINGS[key]))

def save_settings():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        add_system_event("Settings file updated")
    except Exception as e:
        print(f"Error saving settings: {e}")
        add_system_event(f"Settings save failed: {e}", level="error")
        raise

def update_settings_from_payload(payload):
    global settings
    new_settings = DEFAULT_SETTINGS.copy()
    new_settings.update(settings)
    new_settings["bot_token"] = str(payload.get("bot_token", "")).strip()
    new_settings["chat_id"] = str(payload.get("chat_id", "")).strip()
    new_settings["message_cooldown_sec"] = max(0, int(float(payload.get("message_cooldown_sec", settings["message_cooldown_sec"]) or 0)))
    new_settings["fall_confirm_window_sec"] = max(1.0, float(payload.get("fall_confirm_window_sec", settings["fall_confirm_window_sec"]) or 1.0))
    new_settings["preferred_camera"] = str(payload.get("preferred_camera", settings["preferred_camera"]) or "0")
    new_settings["max_people_to_track"] = min(10, max(1, int(float(payload.get("max_people_to_track", settings["max_people_to_track"]) or 1))))
    new_settings["deployment_mode"] = str(payload.get("deployment_mode", settings["deployment_mode"]) or settings["deployment_mode"]).strip().lower()
    if new_settings["deployment_mode"] not in {"server", "edge", "standalone"}:
        new_settings["deployment_mode"] = settings["deployment_mode"]
    new_settings["node_id"] = str(payload.get("node_id", settings["node_id"]) or settings["node_id"]).strip()
    new_settings["central_server_url"] = str(payload.get("central_server_url", settings["central_server_url"]) or settings["central_server_url"]).strip().rstrip("/")
    new_settings["server_bind_host"] = str(payload.get("server_bind_host", settings["server_bind_host"]) or settings["server_bind_host"]).strip()
    new_settings["server_port"] = max(1, int(float(payload.get("server_port", settings["server_port"]) or settings["server_port"])))
    for key in ["enable_telegram", "enable_detection", "enable_voice_alert", "enable_wellness_monitoring", "display_metrics_overlay"]:
        new_settings[key] = bool(payload.get(key, False))
    settings = new_settings
    save_settings()
    add_system_event("Settings saved")

def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["server", "edge", "standalone"])
    parser.add_argument("--node-id")
    parser.add_argument("--central-server-url")
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    args, _ = parser.parse_known_args()
    return args

def apply_runtime_overrides():
    args = parse_runtime_args()
    if args.mode:
        settings["deployment_mode"] = args.mode
    if args.node_id:
        settings["node_id"] = args.node_id.strip()
    if args.central_server_url:
        settings["central_server_url"] = args.central_server_url.strip().rstrip("/")
    if args.server_host:
        settings["server_bind_host"] = args.server_host.strip()
    if args.server_port:
        settings["server_port"] = int(args.server_port)

def is_edge_mode():
    return settings.get("deployment_mode") == "edge"

def is_server_mode():
    return settings.get("deployment_mode") in {"server", "standalone"}

def get_central_server_url():
    return str(settings.get("central_server_url", DEFAULT_SETTINGS["central_server_url"]) or DEFAULT_SETTINGS["central_server_url"]).rstrip("/")

def update_evaluation_metric(key, value):
    with data_lock:
        evaluation_metrics[key] = value

def increment_evaluation_metric(key, amount=1):
    with data_lock:
        evaluation_metrics[key] = evaluation_metrics.get(key, 0) + amount

def record_wake_latency(latency_sec):
    with data_lock:
        evaluation_metrics["wake_events"] = evaluation_metrics.get("wake_events", 0) + 1
        evaluation_metrics["last_wake_latency_sec"] = round(float(latency_sec), 2)
        count = evaluation_metrics["wake_events"]
        current_avg = float(evaluation_metrics.get("avg_wake_latency_sec") or 0.0)
        if count <= 1:
            evaluation_metrics["avg_wake_latency_sec"] = round(float(latency_sec), 2)
        else:
            updated_avg = ((current_avg * (count - 1)) + float(latency_sec)) / count
            evaluation_metrics["avg_wake_latency_sec"] = round(updated_avg, 2)

def get_evaluation_snapshot():
    today = str(date.today())
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT type, COUNT(*) FROM falls WHERE DATE(timestamp)=? GROUP BY type",
            (today,)
        )
        rows = c.fetchall()
        conn.close()
    except Exception:
        rows = []

    fall_counts_today = {row[0]: int(row[1]) for row in rows}
    with data_lock:
        metrics = dict(evaluation_metrics)
        metrics["camera_available"] = camera_available
        metrics["has_frame"] = latest_stream_frame is not None
        metrics["deployment_mode"] = settings.get("deployment_mode")
        metrics["node_id"] = settings.get("node_id")
        metrics["connected_nodes"] = len(remote_nodes)
        metrics["multi_person_mode"] = bool(multi_person_scene_active)
        metrics["visible_people_count"] = int(multi_person_scene_count)

    metrics["date"] = today
    metrics["falls_today"] = int(sum(fall_counts_today.values()))
    metrics["minor_falls_today"] = int(fall_counts_today.get("MINOR FALL", 0))
    metrics["major_falls_today"] = int(fall_counts_today.get("MAJOR FALL", 0))
    metrics["recoveries_today"] = int(fall_counts_today.get("RECOVERED", 0))
    return metrics

def get_public_settings():
    with data_lock:
        data = dict(settings)
    return data

def telegram_ready():
    return bool(settings.get("enable_telegram") and settings.get("bot_token") and settings.get("chat_id"))

def send_telegram_message(message, category="general", force=False, silent=False):
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"
    now = time.time()
    cooldown = int(settings.get("message_cooldown_sec", 90))
    last_sent = last_telegram_sent_at.get(category, 0)
    if not force and cooldown > 0 and (now - last_sent) < cooldown:
        return False, "Cooldown active"
    url = f"https://api.telegram.org/bot{settings['bot_token']}/sendMessage"
    payload = {
        "chat_id": settings["chat_id"],
        "text": message,
        "disable_notification": bool(silent)
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        last_telegram_sent_at[category] = now
        add_system_event(f"Telegram message sent: {category}")
        return True, "Message sent"
    except Exception as e:
        add_system_event(f"Telegram send failed: {e}", level="error")
        return False, str(e)

def send_telegram_burst(message, category, count=1, delay_sec=0.0, force=False):
    """Send repeated Telegram notifications, mainly for urgent alerts."""
    ok_count = 0
    last_result = "Not sent"
    for idx in range(max(1, int(count))):
        burst_category = f"{category}:burst:{idx}" if count > 1 else category
        ok, last_result = send_telegram_message(message, category=burst_category, force=force)
        if ok:
            ok_count += 1
        if idx < count - 1 and delay_sec > 0:
            time.sleep(delay_sec)
    return ok_count > 0, f"Sent {ok_count}/{count} notifications"

def send_telegram_burst_async(message, category, count=1, delay_sec=0.0, force=False):
    threading.Thread(
        target=send_telegram_burst,
        args=(message, category, count, delay_sec, force),
        daemon=True
    ).start()

def build_telegram_dashboard_text():
    summary = get_daily_summary()
    with data_lock:
        active_states = [state for state in person_state.values() if state != "UNKNOWN"]
        current_activity = active_states[0] if active_states else "AWAY"
        recent_fall = fall_events[-1] if fall_events else None
        active_alert_count = len(active_alerts)
        multi_person_mode = bool(multi_person_scene_active)
        visible_people_count = int(multi_person_scene_count)

    if multi_person_mode:
        lines = [
            "ElderlyCare Mini Dashboard",
            "",
            f"Resident: {SINGLE_PERSON_LABEL}",
            f"Multiple people in frame: {visible_people_count}",
            "Activity tracking: Paused",
            "Status: Only fall alerts are active right now.",
            f"Caregiver status: {summary['caregiver_status'].upper()}",
            f"Active alerts: {active_alert_count}",
        ]
    else:
        lines = [
            "ElderlyCare Mini Dashboard",
            "",
            f"Resident: {SINGLE_PERSON_LABEL}",
            f"Current activity: {current_activity}",
            f"Walking: {summary['walking_dur']}",
            f"Standing: {summary['standing_dur']}",
            f"Sitting: {summary['sitting_dur']}",
            f"Sleeping: {summary['sleeping_dur']}",
            f"Monitored today: {summary['monitored_dur']}",
            f"Caregiver status: {summary['caregiver_status'].upper()}",
            f"Active alerts: {active_alert_count}",
        ]

    if recent_fall:
        lines.append(f"Last fall event: {recent_fall.get('type', 'N/A')} at {recent_fall.get('time_str', 'N/A')}")
    else:
        lines.append("Last fall event: None")

    lines.extend([
        "",
        f"Updated: {time.strftime('%H:%M:%S', time.localtime())}"
    ])
    return "\n".join(lines)

def upsert_telegram_dashboard(force_send=False):
    global telegram_dashboard_message_id, telegram_dashboard_last_text, last_telegram_dashboard_update_at
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"

    dashboard_text = build_telegram_dashboard_text()
    now = time.time()
    if not force_send and dashboard_text == telegram_dashboard_last_text and (now - last_telegram_dashboard_update_at) < 20:
        return True, "Dashboard unchanged"

    if telegram_dashboard_message_id is not None:
        url = f"https://api.telegram.org/bot{settings['bot_token']}/editMessageText"
        payload = {
            "chat_id": settings["chat_id"],
            "message_id": telegram_dashboard_message_id,
            "text": dashboard_text,
            "disable_notification": True
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            telegram_dashboard_last_text = dashboard_text
            last_telegram_dashboard_update_at = now
            return True, "Dashboard updated"
        except Exception:
            telegram_dashboard_message_id = None

    url = f"https://api.telegram.org/bot{settings['bot_token']}/sendMessage"
    payload = {
        "chat_id": settings["chat_id"],
        "text": dashboard_text,
        "disable_notification": True
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        telegram_dashboard_message_id = data.get("result", {}).get("message_id")
        telegram_dashboard_last_text = dashboard_text
        last_telegram_dashboard_update_at = now
        add_system_event("Telegram mini dashboard message created")
        return True, "Dashboard created"
    except Exception as e:
        add_system_event(f"Telegram dashboard update failed: {e}", level="error")
        return False, str(e)

def telegram_dashboard_loop():
    while not shutdown_event.is_set():
        time.sleep(15)
        try:
            if telegram_ready():
                upsert_telegram_dashboard()
        except Exception as e:
            add_system_event(f"Telegram dashboard loop error: {e}", level="error")

def notify_activity_change(pid, activity):
    if not settings.get("enable_wellness_monitoring", True):
        return
    if multi_person_scene_active:
        return
    if "FALL" in activity or activity == "RECOVERED":
        return
    prior = last_notified_activity.get(pid)
    if prior == activity:
        return
    last_notified_activity[pid] = activity
    add_system_event(f"{pid} activity changed to {activity}")
    if telegram_ready():
        upsert_telegram_dashboard(force_send=True)

def notify_multi_person_scene():
    global multi_person_notice_sent
    if multi_person_notice_sent:
        return
    multi_person_notice_sent = True
    add_system_event("Multiple people detected. Telegram dashboard switched to multi-person status.")
    if telegram_ready():
        upsert_telegram_dashboard(force_send=True)

load_settings()
apply_runtime_overrides()
add_system_event("Settings loaded")

# ==================== Face Recognition Setup ====================
FACES_DIR = os.path.join(BASE_DIR, "registered_faces")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)

known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition unavailable. Registration and auto-naming are disabled.")
        known_face_encodings = []
        known_face_names = []
        return
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} registered faces.")

load_encodings()

def register_face(image_or_list, name, yolo_model=None):
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    if image_or_list is None: return False
    
    images = image_or_list if isinstance(image_or_list, list) else [image_or_list]
    success_count = 0
    
    for image in images:
        try:
            if isinstance(image, np.ndarray):
                # If we have a YOLO model, try to crop people out first for better accuracy
                if yolo_model:
                    results = yolo_model(image, verbose=False)
                    if results[0].boxes:
                        for box in results[0].boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            crop = image[y1:y2, x1:x2]
                            if process_single_image(crop, name):
                                success_count += 1
                        continue # Already processed crops for this frame
                
                if process_single_image(image, name):
                    success_count += 1
            else: # Flask FileStorage
                file_bytes = np.frombuffer(image.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None: continue
                if process_single_image(img, name):
                    success_count += 1
        except Exception as e:
            print(f"Face Registration Error: {e}")
            
    if success_count > 0:
        with data_lock:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        return True
    return False

def process_single_image(img, name):
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        encodings = face_recognition.face_encodings(rgb, boxes)
        if len(encodings) > 0:
            with data_lock:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            return True
    except: pass
    return False

# ==================== Database Setup ====================
DB_PATH = os.path.join(BASE_DIR, "monitor_data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for daily activity summaries
    c.execute('''CREATE TABLE IF NOT EXISTS activity
                 (date TEXT, person_id TEXT, walking REAL, standing REAL, sitting REAL, sleeping REAL, PRIMARY KEY(date, person_id))''')
    
    # Migration: Add standing column if it doesn't exist
    try:
        c.execute("ALTER TABLE activity ADD COLUMN standing REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Already exists

    # Table for fall events
    c.execute('''CREATE TABLE IF NOT EXISTS falls
                 (timestamp DATETIME, person_id TEXT, type TEXT)''')
    
    # Migration: Add unix_timestamp column if it doesn't exist
    try:
        c.execute("ALTER TABLE falls ADD COLUMN unix_timestamp REAL")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

init_db()

def log_activity_to_db():
    """Sync current in-memory stats to DB and save ReID banks every minute"""
    while not shutdown_event.is_set():
        time.sleep(60)
        # Snapshot the data while holding lock to minimize contention
        with data_lock:
            stats_snapshot = []
            if person_state or any(walking_time.values()) or any(standing_time.values()): # Only if there is active data
                for pid in list(all_tracked_people):
                    stats_snapshot.append((str(pid), walking_time.get(pid, 0), standing_time.get(pid, 0),
                                         sitting_time.get(pid, 0), sleeping_time.get(pid, 0)))
        
        if not stats_snapshot:
            # Just save the banks and continue if no activity stats to log
            with data_lock:
                reid_manager.save_bank()
                save_manual_id_map()
            continue

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            today = str(date.today())
            for pid, w, st, s, sl in stats_snapshot:
                c.execute('''INSERT OR REPLACE INTO activity (date, person_id, walking, standing, sitting, sleeping)
                             VALUES (?, ?, ?, ?, ?, ?)''', (today, pid, w, st, s, sl))
            conn.commit()
            conn.close()
            
            # Save ReID and Identity mapping
            with data_lock:
                protected = list(manual_id_map.keys())
                pruned_ids = reid_manager.prune_bank(protected_ids=protected)
                for pid in pruned_ids:
                    if pid in all_tracked_people:
                        all_tracked_people.remove(pid)
                    # Also remove from stats if they have very little time (ghosts)
                    total_time = walking_time.get(pid, 0) + standing_time.get(pid, 0) + sitting_time.get(pid, 0) + sleeping_time.get(pid, 0)
                    if total_time < 5:
                        walking_time.pop(pid, None)
                        standing_time.pop(pid, None)
                        sitting_time.pop(pid, None)
                        sleeping_time.pop(pid, None)
                
                reid_manager.save_bank()
                save_manual_id_map()
            print("✓ Database and ReID banks synchronized.")
        except Exception as e:
            print(f"Sync Error: {e}")

# Start DB sync thread (MOVED TO END OF INITIALIZATION)

# ==================== Flask Server ====================
app = Flask(__name__)
fall = False
active_alerts = []  # List of unacknowledged falls

# Track fall events with timestamps (history)
fall_events = []

# Track walking/sleeping/sitting/standing durations
walking_time = defaultdict(float)
standing_time = defaultdict(float)
sleeping_time = defaultdict(float)
sitting_time = defaultdict(float)

# Track current state per person
person_state = {}
person_last_time = {}
lying_start_time = {} # Track when a person started lying down
minor_fall_start_time = {} # Track duration of minor fall for escalation
recovery_mode = {} # pid -> expiry_time (suppress minor fall alerts while getting up)
recovery_confirm_count = {} # persistent_id -> count (frames of sustained upright activity)
active_fall_event = {} # pid -> True (prevent multiple alerts for the same fall)

# Movement tracking for static object filtering and activity refinement
person_start_pos = {}
person_last_pos = {} # Track last frame position for velocity
person_velocity = defaultdict(float) # Rolling average velocity
person_vertical_velocity = defaultdict(float) # Rolling average vertical velocity
person_previous_torso_angle = {}
person_previous_torso_time = {}
person_horizontal_stable_start = {}
person_transition_start = {}
person_recent_sudden_drop_until = {}
person_frames_seen = {}
person_is_confirmed = {}

# Squelch logic for ghost IDs/phantom bodies
last_global_alert_time = 0
last_alert_coords = {} # type -> (x, y)
last_alert_pid = {}   # type -> pid
all_tracked_people = set()  # Persistent list of all detected IDs
manual_id_map = {} # Manual link: YOLO_ID -> Registered Name
person_signatures = {} # Store color histograms: Name -> Histogram
guest_walking_time = defaultdict(float)
guest_standing_time = defaultdict(float)
guest_sleeping_time = defaultdict(float)
guest_sitting_time = defaultdict(float)
guest_state = {}
guest_last_time = {}
guest_first_seen = {}
guest_last_seen = {}
multi_person_scene_active = False
multi_person_scene_count = 0

ID_MAP_FILE = os.path.join(BASE_DIR, "manual_id_map.pickle")
WARD_PROFILE_FILE = os.path.join(BASE_DIR, "ward_profile.pickle")
VIDEO_ANALYSIS_UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
VIDEO_ANALYSIS_RESULT_DIR = os.path.join(BASE_DIR, "video_analysis_results")
VIDEO_ANALYSIS_GRAPH_DIR = os.path.join(BASE_DIR, "video_analysis_graphs")
os.makedirs(VIDEO_ANALYSIS_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_ANALYSIS_RESULT_DIR, exist_ok=True)
os.makedirs(VIDEO_ANALYSIS_GRAPH_DIR, exist_ok=True)
video_analysis_jobs = {}

def load_manual_id_map():
    global manual_id_map
    if os.path.exists(ID_MAP_FILE):
        try:
            with open(ID_MAP_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                manual_id_map = data
                print(f"Loaded {len(manual_id_map)} manual ID mappings.")
            else:
                print(f"Warning: {ID_MAP_FILE} had invalid format. Starting fresh.")
                manual_id_map = {}
        except Exception as e:
            print(f"Error loading manual ID map: {e}")
            manual_id_map = {}

def save_manual_id_map():
    try:
        # Prevent overwriting with empty map if it's likely a load failure
        # (Only save if we have mappings or if the file didn't exist)
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(manual_id_map, f)
    except Exception as e:
        print(f"Error saving manual ID map: {e}")

load_manual_id_map()

def load_ward_profile():
    global ward_profile
    if not os.path.exists(WARD_PROFILE_FILE):
        ward_profile = {}
        return
    try:
        with open(WARD_PROFILE_FILE, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            ward_profile = {
                "name": str(data.get("name", "")).strip(),
                "embeddings": [emb for emb in data.get("embeddings", []) if isinstance(emb, np.ndarray)],
                "color_signatures": [sig for sig in data.get("color_signatures", []) if sig is not None],
                "updated_at": float(data.get("updated_at", time.time()) or time.time())
            }
        else:
            ward_profile = {}
    except Exception as e:
        print(f"Error loading ward profile: {e}")
        ward_profile = {}

def save_ward_profile():
    try:
        with open(WARD_PROFILE_FILE, "wb") as f:
            pickle.dump(ward_profile, f)
    except Exception as e:
        print(f"Error saving ward profile: {e}")

def ward_profile_ready():
    return bool(ward_profile.get("name")) and bool(ward_profile.get("embeddings") or ward_profile.get("color_signatures"))

def get_registered_identities():
    names = []
    seen = set()

    def add_name(raw_name):
        clean = str(raw_name or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            names.append(clean)

    add_name(ward_profile.get("name"))
    for value in manual_id_map.values():
        add_name(value)
    for value in known_face_names:
        add_name(value)
    return sorted(names)

def evaluate_full_body_visibility(confidences):
    if confidences is None or len(confidences) < 17:
        return False, "Pose is incomplete. Step back until your full body is visible."
    shoulders_ok = confidences[5] > 0.5 and confidences[6] > 0.5
    hips_ok = confidences[11] > 0.5 and confidences[12] > 0.5
    ankles_ok = confidences[15] > 0.5 and confidences[16] > 0.5
    knees_ok = confidences[13] > 0.4 and confidences[14] > 0.4
    if shoulders_ok and hips_ok and ankles_ok:
        return True, ""
    if not ankles_ok:
        return False, "Step back until both feet and full legs are visible."
    if not hips_ok:
        return False, "Keep your waist and hips clearly visible in the frame."
    if not shoulders_ok:
        return False, "Keep both shoulders visible and face the camera more clearly."
    if not knees_ok:
        return False, "Step back a little more so the full body is visible."
    return False, "Show your full body before starting registration."

def evaluate_registration_capture_visibility(confidences):
    if confidences is None or len(confidences) < 17:
        return False, "Pose is incomplete for registration."

    shoulder_count = int(confidences[5] > 0.25) + int(confidences[6] > 0.25)
    hip_count = int(confidences[11] > 0.3) + int(confidences[12] > 0.3)
    knee_count = int(confidences[13] > 0.25) + int(confidences[14] > 0.25)
    ankle_count = int(confidences[15] > 0.2) + int(confidences[16] > 0.2)

    upper_ok = shoulder_count >= 1 and hip_count >= 1
    lower_ok = ankle_count >= 1 or knee_count >= 2

    if upper_ok and lower_ok:
        return True, ""
    if not lower_ok:
        return False, "Keep more of the legs and feet visible while turning."
    return False, "Keep your torso visible while turning."

def get_registration_visibility_status(target_id):
    meta = latest_person_pose_meta.get(str(target_id))
    if not meta:
        return False, "Stand in front of the camera with your full body visible."
    return bool(meta.get("registration_visible")), str(meta.get("registration_message", "Show your full body before starting registration."))

def make_safe_folder_name(name):
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in str(name or "").strip())
    safe = safe.replace(" ", "_").strip("._")
    return safe or "ward"

def get_person_gallery_dir(name):
    folder = os.path.join(WARD_GALLERY_DIR, make_safe_folder_name(name))
    os.makedirs(folder, exist_ok=True)
    return folder

def delete_registered_identity(name):
    global ward_profile, ward_locked_persistent_id, monitored_persistent_id
    clean_name = str(name or "").strip()
    if not clean_name:
        return False, "Missing name"

    removed_any = False

    gallery_dir = os.path.join(WARD_GALLERY_DIR, make_safe_folder_name(clean_name))
    if os.path.isdir(gallery_dir):
        try:
            shutil.rmtree(gallery_dir)
            removed_any = True
        except Exception as e:
            return False, f"Could not remove gallery: {e}"

    if clean_name in known_face_names:
        filtered_encodings = []
        filtered_names = []
        for enc, existing_name in zip(known_face_encodings, known_face_names):
            if str(existing_name) != clean_name:
                filtered_encodings.append(enc)
                filtered_names.append(existing_name)
        known_face_encodings[:] = filtered_encodings
        known_face_names[:] = filtered_names
        try:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
            removed_any = True
        except Exception as e:
            return False, f"Could not update face encodings: {e}"

    matching_ids = [pid for pid, mapped_name in manual_id_map.items() if str(mapped_name) == clean_name]
    for pid in matching_ids:
        manual_id_map.pop(pid, None)
        if str(ward_locked_persistent_id) == str(pid):
            ward_locked_persistent_id = None
        if str(monitored_persistent_id) == str(pid):
            monitored_persistent_id = None
        removed_any = True
    save_manual_id_map()

    if ward_profile.get("name") == clean_name:
        ward_profile = {}
        save_ward_profile()
        ward_locked_persistent_id = None
        removed_any = True

    if removed_any:
        add_system_event(f"Removed registered identity: {clean_name}")
        return True, f"Removed {clean_name}"
    return False, f"No registered data found for {clean_name}"

def begin_pending_registration(name, preferred_target_id=None):
    pending_registration["active"] = True
    pending_registration["name"] = str(name).strip()
    pending_registration["preferred_target_id"] = str(preferred_target_id) if preferred_target_id else None
    pending_registration["requested_at"] = time.time()
    add_system_event(f"Registration queued for {name}. Waiting for full-body visibility.")

def clear_pending_registration():
    pending_registration["active"] = False
    pending_registration["name"] = ""
    pending_registration["preferred_target_id"] = None
    pending_registration["requested_at"] = 0.0

def add_ward_signature_sample(embedding=None, color_sig=None):
    updated = False
    if embedding is not None:
        emb_list = ward_profile.setdefault("embeddings", [])
        is_new = True
        for existing in emb_list:
            if existing.shape == embedding.shape and np.dot(existing, embedding) > 0.92:
                is_new = False
                break
        if is_new and len(emb_list) < 12:
            emb_list.append(embedding)
            updated = True

    if color_sig is not None:
        sig_list = ward_profile.setdefault("color_signatures", [])
        is_new = True
        for existing in sig_list:
            if compare_signatures(existing, color_sig) > 0.92:
                is_new = False
                break
        if is_new and len(sig_list) < 12:
            sig_list.append(color_sig)
            updated = True

    if updated:
        ward_profile["updated_at"] = time.time()
        save_ward_profile()
    return updated

def start_ward_registration_session(target_id, name):
    clear_pending_registration()
    ward_registration_session["active"] = True
    ward_registration_session["target_id"] = str(target_id)
    ward_registration_session["name"] = str(name).strip()
    ward_registration_session["captures"] = 0
    ward_registration_session["required_captures"] = 220
    ward_registration_session["expires_at"] = time.time() + 32.0
    ward_registration_session["last_capture_at"] = 0.0
    ward_registration_session["started_at"] = time.time()
    ward_registration_session["get_ready_until"] = time.time() + 5.0
    ward_registration_session["current_step"] = 0
    ward_registration_session["steps"] = list(WARD_REGISTRATION_STEPS)
    ward_registration_session["captured_steps"] = []
    ward_registration_session["auto_mode"] = True
    ward_registration_session["capture_interval_sec"] = 0.08
    ward_registration_session["last_saved_index"] = 0
    ward_registration_session["gallery_dir"] = get_person_gallery_dir(name)
    ward_registration_session["completed_at"] = 0.0
    ward_registration_session["completed_name"] = ""
    ward_registration_session["completed_captures"] = 0
    ward_profile["name"] = str(name).strip()
    ward_profile.setdefault("embeddings", [])
    ward_profile.setdefault("color_signatures", [])
    ward_profile["updated_at"] = time.time()
    save_ward_profile()
    add_system_event(f"Ward registration started for {name}. Automatic 360 capture begins in 5 seconds.")

def stop_ward_registration_session(completed=False):
    name = ward_registration_session.get("name", "ward")
    captures = ward_registration_session.get("captures", 0)
    required = ward_registration_session.get("required_captures", 220)
    if ward_registration_session.get("active"):
        if completed:
            add_system_event(
                f"Ward registration completed for {name} with {captures} auto-captured body samples."
            )
            ward_registration_session["completed_at"] = time.time()
            ward_registration_session["completed_name"] = name
            ward_registration_session["completed_captures"] = captures
        else:
            add_system_event(
                f"Ward registration stopped for {name} with {captures}/{required} body-angle samples."
            )
    ward_registration_session["active"] = False
    ward_registration_session["target_id"] = None
    ward_registration_session["name"] = ""
    ward_registration_session["captures"] = 0
    ward_registration_session["required_captures"] = required
    ward_registration_session["expires_at"] = 0.0
    ward_registration_session["last_capture_at"] = 0.0
    ward_registration_session["started_at"] = 0.0
    ward_registration_session["get_ready_until"] = 0.0
    ward_registration_session["current_step"] = 0
    ward_registration_session["steps"] = list(WARD_REGISTRATION_STEPS)
    ward_registration_session["captured_steps"] = []
    ward_registration_session["auto_mode"] = True
    ward_registration_session["capture_interval_sec"] = 0.08
    ward_registration_session["last_saved_index"] = 0
    ward_registration_session["gallery_dir"] = ""

def auto_capture_ward_registration_sample(persistent_id, person_img):
    if not ward_registration_session.get("active"):
        return False
    if str(persistent_id) != str(ward_registration_session.get("target_id")):
        return False
    if person_img is None or getattr(person_img, "size", 0) == 0:
        return False

    now = time.time()
    if now < float(ward_registration_session.get("get_ready_until", 0.0) or 0.0):
        return False
    if now > float(ward_registration_session.get("expires_at", 0.0) or 0.0):
        stop_ward_registration_session(completed=ward_registration_session.get("captures", 0) >= 25)
        return False

    full_body_visible, _ = get_registration_visibility_status(persistent_id)
    if not full_body_visible:
        return False

    min_gap = float(ward_registration_session.get("capture_interval_sec", 0.15) or 0.15)
    if (now - float(ward_registration_session.get("last_capture_at", 0.0) or 0.0)) < min_gap:
        return False

    embedding = reid_manager.get_embedding(person_img)
    color_sig = get_color_signature(person_img)
    added = add_ward_signature_sample(embedding=embedding, color_sig=color_sig)

    gallery_dir = ward_registration_session.get("gallery_dir") or get_person_gallery_dir(ward_registration_session.get("name", "ward"))
    next_index = int(ward_registration_session.get("last_saved_index", 0) or 0) + 1
    file_path = os.path.join(gallery_dir, f"sample_{next_index:04d}.jpg")
    try:
        cv2.imwrite(file_path, person_img)
    except Exception:
        pass

    ward_registration_session["last_saved_index"] = next_index
    ward_registration_session["captures"] += 1
    ward_registration_session["last_capture_at"] = now

    capture_goal = int(ward_registration_session.get("required_captures", 220) or 220)
    if ward_registration_session["captures"] >= capture_goal:
        stop_ward_registration_session(completed=True)
    elif added and ward_registration_session["captures"] >= 80 and now >= (float(ward_registration_session.get("started_at", now) or now) + 24.0):
        stop_ward_registration_session(completed=True)
    return True

def get_ward_registration_status():
    now = time.time()
    completed_recently = bool(
        ward_registration_session.get("completed_at")
        and (now - float(ward_registration_session.get("completed_at") or 0.0)) <= 20.0
    )
    if pending_registration.get("active") and not ward_registration_session.get("active"):
        return {
            "active": True,
            "completed": False,
            "state": "wait_full_body",
            "name": pending_registration.get("name", ""),
            "captures": 0,
            "required_captures": 220,
            "current_step": 0,
            "step": WARD_REGISTRATION_STEPS[0],
            "countdown_sec": 0,
            "captured_steps": []
        }
    if not ward_registration_session.get("active"):
        return {
            "active": False,
            "completed": completed_recently,
            "name": ward_registration_session.get("completed_name", ""),
            "captures": int(ward_registration_session.get("completed_captures", 0)),
            "required_captures": len(WARD_REGISTRATION_STEPS)
        }

    get_ready_until = float(ward_registration_session.get("get_ready_until", 0.0) or 0.0)
    current_step = int(ward_registration_session.get("current_step", 0) or 0)
    steps = ward_registration_session.get("steps", WARD_REGISTRATION_STEPS)
    step = steps[current_step] if current_step < len(steps) else None
    state = "get_ready" if now < get_ready_until else "capture"

    return {
        "active": True,
        "completed": False,
        "state": state,
        "name": ward_registration_session.get("name", ""),
        "captures": int(ward_registration_session.get("captures", 0)),
        "required_captures": int(ward_registration_session.get("required_captures", 220)),
        "current_step": current_step,
        "step": step,
        "countdown_sec": max(0, int(np.ceil((get_ready_until if state == "get_ready" else ward_registration_session.get("expires_at", now)) - now))),
        "captured_steps": list(ward_registration_session.get("captured_steps", [])),
        "auto_mode": bool(ward_registration_session.get("auto_mode", True))
    }

def perform_ward_registration_capture():
    if not ward_registration_session.get("active"):
        return False, "No active ward registration session.", None
    if ward_registration_session.get("auto_mode", True):
        return False, "Automatic 360 capture is running. No manual capture is needed.", None

    now = time.time()
    if now > float(ward_registration_session.get("expires_at", 0.0) or 0.0):
        stop_ward_registration_session(completed=False)
        return False, "Registration session expired. Please start again.", None

    if now < float(ward_registration_session.get("get_ready_until", 0.0) or 0.0):
        remaining = max(1, int(np.ceil(ward_registration_session["get_ready_until"] - now)))
        return False, f"Get ready first. Capture starts in {remaining}s.", None

    current_step = int(ward_registration_session.get("current_step", 0) or 0)
    steps = ward_registration_session.get("steps", WARD_REGISTRATION_STEPS)
    if current_step >= len(steps):
        stop_ward_registration_session(completed=True)
        return True, "All angle captures are already complete.", None

    target_id = str(ward_registration_session.get("target_id") or "")
    person_img = latest_person_crops.get(target_id)
    if person_img is None or getattr(person_img, "size", 0) == 0:
        return False, "Ward body is not clearly visible. Stand fully in frame and try again.", steps[current_step]
    full_body_visible, visibility_message = get_registration_visibility_status(target_id)
    if not full_body_visible:
        return False, visibility_message, steps[current_step]

    embedding = reid_manager.get_embedding(person_img)
    color_sig = get_color_signature(person_img)
    add_ward_signature_sample(embedding=embedding, color_sig=color_sig)

    step = steps[current_step]
    ward_registration_session["captures"] = current_step + 1
    ward_registration_session["last_capture_at"] = now
    ward_registration_session["captured_steps"].append(step["code"])
    ward_registration_session["current_step"] = current_step + 1

    if ward_registration_session["current_step"] >= len(steps):
        ward_registration_session["captures"] = len(steps)
        stop_ward_registration_session(completed=True)
        return True, f"Captured {step['label']}. All required angles have been saved for identity persistence.", step

    next_step = steps[ward_registration_session["current_step"]]
    return True, f"Captured {step['label']}. Next: {next_step['label']}.", step

def match_ward_profile(person_img):
    if not ward_profile_ready() or person_img is None or person_img.size == 0:
        return False, 0.0

    best_score = 0.0
    embedding = reid_manager.get_embedding(person_img)
    if embedding is not None:
        for existing in ward_profile.get("embeddings", []):
            if existing.shape == embedding.shape:
                best_score = max(best_score, float(np.dot(existing, embedding)))
        if best_score >= 0.72:
            return True, best_score

    color_sig = get_color_signature(person_img)
    if color_sig is not None:
        for existing in ward_profile.get("color_signatures", []):
            best_score = max(best_score, float(compare_signatures(existing, color_sig)))
        if best_score >= 0.75:
            return True, best_score

    return False, best_score

load_ward_profile()

def get_display_id(persistent_id):
    """Resolve how a tracked person should appear in stats and the dashboard."""
    if persistent_id in manual_id_map:
        return manual_id_map[persistent_id]
    if SINGLE_PERSON_MODE and ward_profile.get("name"):
        return ward_profile.get("name")
    if SINGLE_PERSON_MODE:
        return SINGLE_PERSON_LABEL
    return persistent_id

def get_primary_monitored_id():
    if ward_locked_persistent_id:
        return str(ward_locked_persistent_id)
    ward_name = str(ward_profile.get("name", "") or "").strip()
    if ward_name:
        for pid, mapped_name in manual_id_map.items():
            if str(mapped_name) == ward_name:
                return str(pid)
    if monitored_persistent_id:
        return str(monitored_persistent_id)
    return None

def get_guest_display_name(persistent_id):
    if persistent_id in manual_id_map:
        return f"{manual_id_map[persistent_id]} (Guest)"
    return f"Guest {persistent_id}"

def get_color_signature(image):
    """Calculate color histogram for Re-Identification"""
    try:
        if image is None or image.size == 0: return None
        # Focus on the torso (center of the crop)
        h, w = image.shape[:2]
        torso = image[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.8)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except: return None

def compare_signatures(sig1, sig2):
    """Compare two color histograms (0.0 to 1.0)"""
    if sig1 is None or sig2 is None: return 0
    return cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)

def get_pose_confidence(confidences):
    if confidences is None or len(confidences) < 17:
        return 0.0
    critical_joints = [5, 6, 11, 12, 15, 16]
    return float(sum(confidences[j] for j in critical_joints) / len(critical_joints))

def draw_ward_registration_overlay(frame):
    if frame is None:
        return

    now = time.time()
    if ward_registration_session.get("active"):
        get_ready_until = float(ward_registration_session.get("get_ready_until", 0.0) or 0.0)
        captures = int(ward_registration_session.get("captures", 0))
        required = int(ward_registration_session.get("required_captures", len(WARD_REGISTRATION_STEPS)))
        name = ward_registration_session.get("name", "ward")
        current_step = int(ward_registration_session.get("current_step", 0) or 0)
        steps = ward_registration_session.get("steps", WARD_REGISTRATION_STEPS)
        step = steps[current_step] if current_step < len(steps) else None

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), (20, 110, 230), -1)
        cv2.putText(
            frame,
            f"Registering ward profile for {name}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            "Stand back and slowly do a full 360 turn. Capture is automatic.",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )
        if now < get_ready_until:
            remaining = max(0, int(np.ceil(get_ready_until - now)))
            line = f"Get ready: hold still, full body visible. Starting in {remaining}s"
        else:
            remaining = max(0, int(np.ceil(ward_registration_session.get("expires_at", now) - now)))
            line = f"Auto capture running: turn slowly | Saved {captures} samples | {remaining}s left"
        cv2.putText(
            frame,
            f"{line}    Captures: {captures}/{required}",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2
        )
        return

    completed_at = float(ward_registration_session.get("completed_at", 0.0) or 0.0)
    if completed_at and (now - completed_at) <= 6.0:
        completed_name = ward_registration_session.get("completed_name", "ward")
        completed_captures = int(ward_registration_session.get("completed_captures", 0))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 74), (20, 170, 70), -1)
        cv2.putText(
            frame,
            f"Registration complete for {completed_name}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Captured {completed_captures} body-angle samples for identity persistence.",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

status_message = ""
status_expiry = 0
camera_available = False

def load_stats_from_db():
    global walking_time, standing_time, sleeping_time, sitting_time, all_tracked_people
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        today = str(date.today())
        c.execute("SELECT person_id, walking, standing, sitting, sleeping FROM activity WHERE date=?", (today,))
        rows = c.fetchall()
        with data_lock:
            for r in rows:
                pid, w, st, s, sl = r
                display_id = SINGLE_PERSON_LABEL if SINGLE_PERSON_MODE else pid
                walking_time[display_id] += w
                standing_time[display_id] += st
                sitting_time[display_id] += s
                sleeping_time[display_id] += sl
                all_tracked_people.add(display_id)
        conn.close()
        print(f"Loaded stats for {len(rows)} people from database.")
    except Exception as e:
        print(f"Error loading stats: {e}")

load_stats_from_db()

def load_fall_history():
    global fall_events
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, person_id, type, unix_timestamp FROM falls ORDER BY unix_timestamp DESC LIMIT 50")
        rows = c.fetchall()
        for r in rows:
            ts, pid, ftype, uts = r
            
            # Robust timestamp parsing
            if isinstance(ts, str):
                try:
                    # SQLite datetime.now() usually looks like '2026-01-28 14:25:10.123456'
                    # or '2026-01-28 14:25:10'
                    dt_obj = datetime.strptime(ts.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    display_time = dt_obj.strftime("%H:%M:%S")
                    derived_uts = dt_obj.timestamp()
                except:
                    display_time = ts # Fallback
                    derived_uts = uts if uts else time.time()
            else:
                display_time = ts.strftime("%H:%M:%S")
                derived_uts = uts if uts else ts.timestamp()

            with data_lock:
                fall_events.append({
                    "person": SINGLE_PERSON_LABEL if SINGLE_PERSON_MODE else pid,
                    "type": ftype,
                    "timestamp": uts if uts else derived_uts,
                    "time_str": display_time
                })
        # Keep them in chronological order for the list (appends happened in reverse order from SELECT)
        with data_lock:
            fall_events.sort(key=lambda x: x['timestamp'])
        conn.close()
        print(f"Loaded {len(fall_events)} fall events from history.")
    except Exception as e:
        print(f"Error loading fall history: {e}")

load_fall_history()

# Start DB sync thread (Wait for all initializations to complete)
threading.Thread(target=log_activity_to_db, daemon=True).start()
threading.Thread(target=telegram_dashboard_loop, daemon=True).start()

@app.route("/trigger", methods=["POST"])
def trigger():
    global fall, active_alerts
    data = request.get_json(silent=True) or {}
    person_id = data.get("person_id")
    msg = data.get("message", "Unknown")
    fall_type = data.get("type", "FALL")
    
    if not person_id:
        # Backward compatibility or fallback
        person_id = msg.split(' (')[0] if ' (' in msg else msg

    now = time.time()
    
    with data_lock:
        # Update existing alert for this person to prevent duplicates (e.g. Minor -> Major)
        found = False
        for alert in active_alerts:
            # Match by the clean person_id
            if str(alert['person_id']) == str(person_id):
                alert['type'] = fall_type
                alert['message'] = msg
                alert['time_str'] = time.strftime("%H:%M:%S", time.localtime(now))
                alert['timestamp'] = now
                found = True
                break
        
        if not found:
            active_alerts.append({
                "person_id": person_id,
                "message": msg,
                "type": fall_type,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "timestamp": now
            })
    
    fall = True
    return "OK"

@app.route("/api/acknowledge/<pid>", methods=["POST"])
def acknowledge(pid):
    global active_alerts
    with data_lock:
        # Match by person_id for precise removal
        active_alerts = [a for a in active_alerts if str(a['person_id']) != str(pid) and str(a['message']) != str(pid)]
    return jsonify({"status": "success"})

@app.route("/fall")
def check():
    with data_lock:
        is_fall = len(active_alerts) > 0
    return jsonify({"fall": is_fall})

last_frame = None

def rename_person(old_id, new_name):
    """Safely rename a person and transfer all their stats and mappings."""
    with data_lock:
        _rename_person_internal(old_id, new_name)

def _rename_person_internal(old_id, new_name):
    # 1. Resolve to actual persistent_id (Person_N) if old_id is already a name
    target_persistent_id = old_id
    for k, v in manual_id_map.items():
        if v == old_id:
            target_persistent_id = k
            break
            
    manual_id_map[str(target_persistent_id)] = new_name
    
    # Transfer current stats (from both old_id and target_persistent_id to new_name)
    for source_id in [old_id, target_persistent_id]:
        if source_id != new_name:
            walking_time[new_name] += walking_time.pop(source_id, 0)
            standing_time[new_name] += standing_time.pop(source_id, 0)
            sitting_time[new_name] += sitting_time.pop(source_id, 0)
            sleeping_time[new_name] += sleeping_time.pop(source_id, 0)
    
    # Update set of all people
    if str(old_id) in all_tracked_people and str(old_id) != new_name:
        all_tracked_people.remove(old_id)
    if str(target_persistent_id) in all_tracked_people and str(target_persistent_id) != new_name:
        all_tracked_people.remove(target_persistent_id)
    all_tracked_people.add(new_name)
    
    # Update active alerts and history in memory
    for alert in active_alerts:
        if str(alert.get('person_id')) == str(old_id) or str(alert.get('person_id')) == str(target_persistent_id):
            alert['person_id'] = new_name
            alert['message'] = alert['message'].replace(str(old_id), new_name).replace(str(target_persistent_id), new_name)
            
    for event in fall_events:
        if str(event.get('person')) == str(old_id) or str(event.get('person')) == str(target_persistent_id):
            event['person'] = new_name

    # Save mappings AND bank immediately to ensure persistence
    save_manual_id_map()
    reid_manager.save_bank()
    
    # Update database: Delete old entries (stats were transferred to new name)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM activity WHERE person_id=?", (str(old_id),))
        if target_persistent_id != old_id:
            c.execute("DELETE FROM activity WHERE person_id=?", (str(target_persistent_id),))
        
        # Also update fall history to link to the new name
        c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(old_id)))
        if target_persistent_id != old_id:
            c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(target_persistent_id)))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating DB for rename: {e}")
        
    print(f"👤 Renamed {old_id} (ID: {target_persistent_id}) to {new_name} and saved.")

@app.route("/register", methods=["POST"])
def register():
    global last_frame, status_message, status_expiry, ward_locked_persistent_id, monitored_persistent_id
    name = request.form.get("name")
    target_id = request.form.get("yolo_id") # Register by active ID (can be persistent_id)
    
    if not name:
        return jsonify({"status": "error", "message": "Missing name"})
    
    start_ward_capture_for = None
    
    # CASE 1: Manual ID naming (No face needed)
    with data_lock:
        if target_id and (str(target_id) in person_state or str(target_id) in all_tracked_people):
            _rename_person_internal(str(target_id), name)
            ward_locked_persistent_id = str(target_id)
            monitored_persistent_id = str(target_id)
            start_ward_capture_for = str(target_id)
            status_message = f"ID {target_id} is now {name}"
            status_expiry = time.time() + 5
        elif not target_id:
            visible_ids = list(person_state.keys())
            unnamed = [pid for pid in visible_ids if pid not in manual_id_map]
            if len(unnamed) == 1:
                target_id = unnamed[0]
                _rename_person_internal(str(target_id), name)
                ward_locked_persistent_id = str(target_id)
                monitored_persistent_id = str(target_id)
                start_ward_capture_for = str(target_id)
                status_message = f"Auto-linked {name} to ID {target_id}"
                status_expiry = time.time() + 5
            elif len(visible_ids) == 1:
                target_id = visible_ids[0]
                _rename_person_internal(str(target_id), name)
                ward_locked_persistent_id = str(target_id)
                monitored_persistent_id = str(target_id)
                start_ward_capture_for = str(target_id)
                status_message = f"Auto-linked visible body {target_id} to {name}"
                status_expiry = time.time() + 5
            elif len(unnamed) > 1:
                 return jsonify({"status": "error", "message": "Multiple unnamed bodies. Click the ID button next to the body instead."})

    if start_ward_capture_for:
        full_body_visible, visibility_message = get_registration_visibility_status(start_ward_capture_for)
        if not full_body_visible:
            begin_pending_registration(name, preferred_target_id=start_ward_capture_for)
            status_message = visibility_message
            status_expiry = time.time() + 5
            return jsonify({
                "status": "success",
                "message": f"{visibility_message} Registration is queued and will start automatically once your full body is visible.",
                "guided_registration": True,
                "waiting_full_body": True
            })
        start_ward_registration_session(start_ward_capture_for, name)
        status_message = f"Registering {name}: get ready, automatic 360 capture starts in 5 seconds"
        status_expiry = time.time() + 6
        return jsonify({
            "status": "success",
            "message": f"Successfully named body {start_ward_capture_for} as {name}. Get ready first, then slowly do a full 360 turn while snapshots are captured automatically.",
            "guided_registration": True
        })

    # CASE 2: Uploaded files or Live frame
    front_img = request.files.get("front")
    back_img = request.files.get("back")

    if not front_img and not back_img and not target_id:
        guidance_message = "Step back and keep your full body visible before starting registration."
        begin_pending_registration(name)
        status_message = guidance_message
        status_expiry = time.time() + 5
        return jsonify({
            "status": "success",
            "message": f"{guidance_message} Registration is queued and will continue automatically.",
            "guided_registration": True,
            "waiting_full_body": True
        })
    
    to_process = []
    if front_img: to_process.append(front_img)
    if back_img: to_process.append(back_img)
    
    if not to_process:
        if last_frame is not None:
            to_process = [last_frame]
        else:
            return jsonify({"status": "error", "message": "No photos uploaded or live frame available"})
    
    body_sample_count = 0
    body_images = []
    for item in to_process:
        if isinstance(item, np.ndarray):
            body_images.append(item)
        else:
            try:
                item.stream.seek(0)
            except Exception:
                pass
            file_bytes = np.frombuffer(item.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                body_images.append(img)
            try:
                item.stream.seek(0)
            except Exception:
                pass

    if body_images:
        for img in body_images:
            samples = []
            try:
                if img is not None:
                    results = model(img, verbose=False)
                    if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                        for box in results[0].boxes.xyxy[:2]:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                            if crop is not None and crop.size > 0:
                                samples.append(crop)
                    else:
                        samples.append(img)
            except Exception:
                samples.append(img)

            for sample in samples:
                emb = reid_manager.get_embedding(sample)
                sig = get_color_signature(sample)
                ward_profile["name"] = str(name).strip()
                if add_ward_signature_sample(embedding=emb, color_sig=sig):
                    body_sample_count += 1

    face_registered = register_face(to_process, name, yolo_model=model if not (front_img or back_img) else None)
    if face_registered or body_sample_count > 0:
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        extra = f" Captured {body_sample_count} body-angle sample(s)." if body_sample_count else ""
        if not face_registered and body_sample_count > 0:
            extra += " Face registration was skipped, but body registration succeeded."
        return jsonify({"status": "success", "message": f"Successfully registered {name}.{extra}"})
    return jsonify({"status": "error", "message": "No face detected in provided images"})

@app.route("/api/history")
def activity_history():
    """Get history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if SINGLE_PERSON_MODE:
        c.execute("""SELECT date, ? as person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping)
                     FROM activity
                     GROUP BY date
                     ORDER BY date DESC LIMIT 50""", (SINGLE_PERSON_LABEL,))
    else:
        c.execute("SELECT date, person_id, walking, standing, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"date": r[0], "pid": r[1], "walk": r[2], "stand": r[3], "sit": r[4], "sleep": r[5]} for r in rows])

@app.route("/api/history/monthly")
def monthly_history():
    """Get monthly aggregated history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if SINGLE_PERSON_MODE:
        c.execute("""SELECT SUBSTR(date, 1, 7) as month, ? as person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping) 
                     FROM activity
                     GROUP BY month
                     ORDER BY month DESC LIMIT 24""", (SINGLE_PERSON_LABEL,))
    else:
        c.execute("""SELECT SUBSTR(date, 1, 7) as month, person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping) 
                     FROM activity 
                     GROUP BY month, person_id 
                     ORDER BY month DESC LIMIT 24""")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"date": r[0], "pid": r[1], "walk": r[2], "stand": r[3], "sit": r[4], "sleep": r[5]} for r in rows])

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        try:
            payload = request.get_json(silent=True) or {}
            update_settings_from_payload(payload)
            return jsonify({"status": "success", "settings": get_public_settings()})
        except Exception as e:
            add_system_event(f"Settings API error: {e}", level="error")
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify(get_public_settings())

@app.route("/api/settings/test-telegram", methods=["POST"])
def api_test_telegram():
    ok, message = send_telegram_message("Test alert from ElderlyCare Pro settings page.", category="test", force=True)
    status = "success" if ok else "error"
    return jsonify({"status": status, "message": message})

@app.route("/api/events")
def api_events():
    with data_lock:
        events = list(reversed(system_events[-30:]))
    return jsonify(events)

@app.route("/api/events/clear", methods=["POST"])
def api_clear_events():
    with data_lock:
        system_events.clear()
    add_system_event("System events cleared")
    return jsonify({"status": "success"})

def generate_video_stream():
    while True:
        with data_lock:
            frame_bytes = latest_stream_frame
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera-status")
def api_camera_status():
    with data_lock:
        has_frame = latest_stream_frame is not None
    return jsonify({"camera_available": camera_available, "has_frame": has_frame})

@app.route("/api/node-heartbeat", methods=["POST"])
def api_node_heartbeat():
    data = request.get_json(silent=True) or {}
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        return jsonify({"status": "error", "message": "Missing node_id"}), 400

    with data_lock:
        remote_nodes[node_id] = {
            "node_id": node_id,
            "camera_available": bool(data.get("camera_available", False)),
            "has_frame": bool(data.get("has_frame", False)),
            "deployment_mode": str(data.get("deployment_mode", "edge")),
            "address": request.remote_addr,
            "last_seen": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime())
        }
    return jsonify({"status": "success"})

@app.route("/api/edge/report", methods=["POST"])
def api_edge_report():
    data = request.get_json(silent=True) or {}
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        return jsonify({"status": "error", "message": "Missing node_id"}), 400

    now = time.time()
    with data_lock:
        remote_edge_reports[node_id] = {
            "node_id": node_id,
            "people": list(data.get("people", [])),
            "falls": list(data.get("falls", [])),
            "active_alerts": list(data.get("active_alerts", [])),
            "unnamed_ids": list(data.get("unnamed_ids", [])),
            "updated_at": now
        }
        remote_nodes[node_id] = {
            "node_id": node_id,
            "camera_available": bool(data.get("camera_available", False)),
            "has_frame": bool(data.get("has_frame", False)),
            "deployment_mode": "edge",
            "address": request.remote_addr,
            "last_seen": now,
            "time_str": time.strftime("%H:%M:%S", time.localtime(now))
        }
    return jsonify({"status": "success"})

@app.route("/api/nodes")
def api_nodes():
    now = time.time()
    with data_lock:
        nodes = [{
            **node,
            "seconds_since_seen": round(now - node["last_seen"], 1),
            "is_online": (now - node["last_seen"]) <= 10.0,
            "people_count": len(remote_edge_reports.get(node["node_id"], {}).get("people", [])),
            "active_alert_count": len(remote_edge_reports.get(node["node_id"], {}).get("active_alerts", [])),
            "last_report_age_sec": round(now - remote_edge_reports.get(node["node_id"], {}).get("updated_at", node["last_seen"]), 1)
        } for node in remote_nodes.values()]
    nodes.sort(key=lambda item: item["node_id"])
    return jsonify(nodes)

@app.route("/settings")
def settings_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>System Settings</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #070b1c;
                --panel: #0d142b;
                --panel-border: #1b2442;
                --text: #f1f5f9;
                --muted: #94a3b8;
                --accent: #1d4ed8;
                --accent-soft: #123454;
                --success: #16a34a;
            }
            body { background: radial-gradient(circle at top left, #0e1533, var(--bg) 45%); color: var(--text); min-height: 100vh; font-family: 'Segoe UI', sans-serif; }
            .shell { max-width: 1240px; margin: 0 auto; padding: 28px 22px 44px; }
            .title { font-size: 2.2rem; font-weight: 800; }
            .subtitle { color: var(--muted); margin-bottom: 28px; }
            .panel { background: rgba(13, 20, 43, 0.95); border: 1px solid var(--panel-border); border-radius: 28px; box-shadow: 0 20px 60px rgba(0,0,0,0.25); }
            .main-panel { padding: 22px; }
            .side-panel { padding: 22px; }
            .form-label { color: #cbd5e1; font-weight: 600; margin-bottom: 8px; }
            .form-control, .form-select { background: #0a1024; border: 1px solid #202b4a; color: var(--text); border-radius: 16px; padding: 14px 16px; }
            .form-control:focus, .form-select:focus { background: #0a1024; color: var(--text); border-color: #3558c8; box-shadow: 0 0 0 .2rem rgba(29,78,216,.2); }
            .setting-check { background: #0a1024; border: 1px solid #202b4a; border-radius: 18px; padding: 16px 18px; }
            .btn-primary { background: linear-gradient(135deg, #18498e, #10365f); border: none; border-radius: 16px; padding: 14px 24px; font-weight: 700; }
            .btn-outline-light { border-color: #253457; color: #e2e8f0; border-radius: 16px; }
            .tool-btn { width: 100%; text-align: left; margin-bottom: 12px; padding: 14px 16px; background: #0a1024; border: 1px solid #202b4a; color: var(--text); border-radius: 16px; }
            .events-box { background: #0a1024; border: 1px solid #202b4a; border-radius: 18px; min-height: 260px; max-height: 340px; overflow-y: auto; padding: 14px; }
            .event-item { border-bottom: 1px solid #1b2442; padding: 10px 0; }
            .event-item:last-child { border-bottom: none; }
            .event-time { color: var(--muted); font-size: 0.8rem; }
            .helper { color: var(--muted); font-size: 0.92rem; }
            .status-pill { display:inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(22,163,74,0.15); color: #86efac; font-size: 0.82rem; }
            a.top-link { color: #bfdbfe; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="d-flex justify-content-between align-items-start flex-wrap gap-3 mb-4">
                <div>
                    <div class="title">System Settings</div>
                    <div class="subtitle">Configuration for alerts, detection, and wellness monitoring.</div>
                </div>
                <div class="d-flex gap-2 align-items-center">
                    <span id="save-state" class="status-pill">Ready</span>
                    <a class="top-link" href="/">Back to Dashboard</a>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-lg-8">
                    <div class="panel main-panel">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Bot Token</label>
                                <input id="bot_token" class="form-control" type="password" placeholder="Telegram bot token">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Chat ID</label>
                                <input id="chat_id" class="form-control" type="text" placeholder="Telegram chat ID">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Message Cooldown (sec)</label>
                                <input id="message_cooldown_sec" class="form-control" type="number" min="0">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Fall Confirm Window (sec)</label>
                                <input id="fall_confirm_window_sec" class="form-control" type="number" min="1" step="0.5">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Preferred Camera</label>
                                <select id="preferred_camera" class="form-select">
                                    <option value="0">Auto / Camera 0</option>
                                    <option value="1">Camera 1</option>
                                    <option value="2">Camera 2</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Max People to Track</label>
                                <input id="max_people_to_track" class="form-control" type="number" min="1" max="10">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Deployment Mode</label>
                                <select id="deployment_mode" class="form-select">
                                    <option value="server">Server</option>
                                    <option value="edge">Edge</option>
                                    <option value="standalone">Standalone</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Node ID</label>
                                <input id="node_id" class="form-control" type="text" placeholder="living-room-laptop">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Central Server URL</label>
                                <input id="central_server_url" class="form-control" type="text" placeholder="http://192.168.1.10:5000">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Server Bind Host</label>
                                <input id="server_bind_host" class="form-control" type="text" placeholder="0.0.0.0">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Server Port</label>
                                <input id="server_port" class="form-control" type="number" min="1" max="65535">
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_telegram" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Telegram</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_detection" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Detection</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_voice_alert" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Voice Alert</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_wellness_monitoring" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Wellness Monitoring</span>
                                </label>
                            </div>
                            <div class="col-12">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="display_metrics_overlay" class="form-check-input mt-0" type="checkbox">
                                    <span>Display Metrics Overlay (on video)</span>
                                </label>
                            </div>
                            <div class="col-12 d-flex align-items-center gap-3 pt-2">
                                <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
                                <span class="helper">Use BotFather to create the bot, then paste your bot token and target chat ID here.</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4">
                    <div class="panel side-panel mb-4">
                        <h4 class="mb-3">Tools</h4>
                        <button class="tool-btn" onclick="sendTestAlert()"><i class="fas fa-paper-plane me-2"></i>Send Test Telegram Alert</button>
                        <button class="tool-btn" onclick="clearLogs()"><i class="fas fa-trash me-2"></i>Clear Logs</button>
                    </div>
                    <div class="panel side-panel">
                        <h4 class="mb-3">System Events</h4>
                        <div id="events-box" class="events-box"><div class="helper">No events yet.</div></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const fields = [
                'bot_token', 'chat_id', 'message_cooldown_sec', 'fall_confirm_window_sec',
                'preferred_camera', 'max_people_to_track', 'deployment_mode', 'node_id',
                'central_server_url', 'server_bind_host', 'server_port', 'enable_telegram',
                'enable_detection', 'enable_voice_alert', 'enable_wellness_monitoring',
                'display_metrics_overlay'
            ];

            function setSaveState(text, good=true) {
                const el = document.getElementById('save-state');
                el.textContent = text;
                el.style.background = good ? 'rgba(22,163,74,0.15)' : 'rgba(220,38,38,0.15)';
                el.style.color = good ? '#86efac' : '#fca5a5';
            }

            function applySettings(data) {
                fields.forEach(id => {
                    const el = document.getElementById(id);
                    if (!el) return;
                    if (el.type === 'checkbox') el.checked = !!data[id];
                    else el.value = data[id] ?? '';
                });
            }

            async function loadSettings() {
                try {
                    setSaveState('Loading...');
                    const res = await fetch('/api/settings');
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Failed to load settings');
                    applySettings(data);
                    setSaveState('Loaded');
                } catch (err) {
                    console.error(err);
                    setSaveState('Load Failed', false);
                    alert(`Could not load settings: ${err.message}`);
                }
            }

            function collectSettings() {
                const payload = {};
                fields.forEach(id => {
                    const el = document.getElementById(id);
                    payload[id] = el.type === 'checkbox' ? el.checked : el.value;
                });
                return payload;
            }

            async function saveSettings() {
                try {
                    setSaveState('Saving...');
                    const res = await fetch('/api/settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(collectSettings())
                    });
                    const data = await res.json();
                    if (!res.ok || data.status !== 'success') {
                        throw new Error(data.message || 'Failed to save settings');
                    }
                    applySettings(data.settings || collectSettings());
                    setSaveState('Saved');
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                    setSaveState('Save Failed', false);
                    alert(`Could not save settings: ${err.message}`);
                }
            }

            async function sendTestAlert() {
                try {
                    const res = await fetch('/api/settings/test-telegram', { method: 'POST' });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Test alert failed');
                    alert(data.message);
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                    alert(`Could not send test alert: ${err.message}`);
                }
            }

            async function clearLogs() {
                try {
                    await fetch('/api/events/clear', { method: 'POST' });
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                }
            }

            async function refreshEvents() {
                const box = document.getElementById('events-box');
                try {
                    const res = await fetch('/api/events');
                    const events = await res.json();
                    if (!res.ok) throw new Error('Failed to load events');
                    if (!events.length) {
                        box.innerHTML = '<div class="helper">No events yet.</div>';
                        return;
                    }
                    box.innerHTML = events.map(e => `
                        <div class="event-item">
                            <div>${e.message}</div>
                            <div class="event-time">${e.time_str} • ${e.level}</div>
                        </div>
                    `).join('');
                } catch (err) {
                    console.error(err);
                    box.innerHTML = '<div class="helper">Unable to load events.</div>';
                }
            }

            loadSettings();
            refreshEvents();
            setInterval(refreshEvents, 5000);
        </script>
    </body>
    </html>
    """
    return html

@app.route("/")
def home():
    """Enhanced modern dashboard with Monthly Analytics and Fall Messaging"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elderly Monitor Pro</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --danger-color: #ef233c;
                --success-color: #2ecc71;
                --warning-color: #f39c12;
                --page-bg: #f0f2f5;
                --text-color: #333333;
                --muted-text: #6c757d;
                --navbar-bg: #ffffff;
                --navbar-border: #e0e0e0;
                --sidebar-bg: #f8f9fa;
                --sidebar-border: #e0e0e0;
                --card-bg: #ffffff;
                --card-shadow: 0 4px 6px rgba(0,0,0,0.05);
                --soft-bg: #f8fafc;
                --panel-bg: #ffffff;
                --border-soft: #e5e7eb;
                --nav-link-color: #666666;
                --nav-link-active-bg: #ffffff;
                --nav-link-active-shadow: 0 2px 4px rgba(0,0,0,0.05);
                --badge-bg: #f8f9fa;
                --badge-text: #212529;
                --badge-border: #dee2e6;
                --input-bg: #ffffff;
                --input-text: #212529;
                --input-border: #dee2e6;
                --list-border: #e9ecef;
            }
            body.theme-dark {
                --page-bg: #0f172a;
                --text-color: #e5edf7;
                --muted-text: #94a3b8;
                --navbar-bg: #111827;
                --navbar-border: #1f2937;
                --sidebar-bg: #0b1220;
                --sidebar-border: #1f2937;
                --card-bg: #111827;
                --card-shadow: 0 12px 28px rgba(0,0,0,0.35);
                --soft-bg: #172033;
                --panel-bg: #0f172a;
                --border-soft: #243041;
                --nav-link-color: #cbd5e1;
                --nav-link-active-bg: #162033;
                --nav-link-active-shadow: 0 8px 18px rgba(0,0,0,0.2);
                --badge-bg: #172033;
                --badge-text: #e5edf7;
                --badge-border: #334155;
                --input-bg: #0f172a;
                --input-text: #e5edf7;
                --input-border: #334155;
                --list-border: #243041;
            }
            body { background-color: var(--page-bg); font-family: 'Inter', sans-serif; color: var(--text-color); transition: background-color .25s ease, color .25s ease; }
            .navbar { background: var(--navbar-bg); border-bottom: 1px solid var(--navbar-border); box-shadow: 0 2px 4px rgba(0,0,0,0.02); transition: background-color .25s ease, border-color .25s ease; }
            .sidebar { background: var(--sidebar-bg); border-right: 1px solid var(--sidebar-border); min-height: 100vh; padding: 20px; transition: background-color .25s ease, border-color .25s ease; }
            .card { background: var(--card-bg); color: var(--text-color); border: none; border-radius: 12px; box-shadow: var(--card-shadow); margin-bottom: 24px; transition: transform 0.2s, background-color .25s ease, color .25s ease, box-shadow .25s ease; }
            .card:hover { transform: translateY(-2px); }
            .status-badge { padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
            
            .bg-walking { background-color: #d1fae5; color: #065f46; }
            .bg-standing { background-color: #dbeafe; color: #1e40af; }
            .bg-sleeping { background-color: #f3e8ff; color: #5b21b6; }
            .bg-sitting { background-color: #fef3c7; color: #92400e; }
            .bg-major-fall { background-color: #fee2e2; color: #991b1b; animation: pulse 2s infinite; }
            .bg-minor-fall { background-color: #ffedd5; color: #9a3412; }
            .bg-recovered { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
            .bg-away { background-color: #f3f4f6; color: #4b5563; }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(239, 35, 60, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0); }
            }

            .alert-item { 
                background: white; border-radius: 10px; padding: 16px; margin-bottom: 12px; 
                border-left: 6px solid var(--danger-color); box-shadow: 0 2px 8px rgba(239, 35, 60, 0.1);
            }
            .alert-item.recovered { border-left-color: var(--success-color); box-shadow: 0 2px 8px rgba(46, 204, 113, 0.1); }
            
            .stat-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
            .icon-walk { background: #d1fae5; color: #10b981; }
            .icon-stand { background: #dbeafe; color: #3b82f6; }
            .icon-sit { background: #fef3c7; color: #f59e0b; }
            .icon-sleep { background: #f3e8ff; color: #8b5cf6; }

            #fall-message-display { position: fixed; top: 20px; right: 20px; z-index: 9999; width: 320px; }
            .toast-fall { 
                background: var(--danger-color); color: white; padding: 16px; border-radius: 8px; margin-bottom: 10px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); display: flex; align-items: center; animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
            
            .nav-link { color: var(--nav-link-color); padding: 10px 15px; border-radius: 8px; margin-bottom: 5px; font-weight: 500; transition: background-color .25s ease, color .25s ease, box-shadow .25s ease; }
            .nav-link:hover, .nav-link.active { background: var(--nav-link-active-bg); color: var(--primary-color); box-shadow: var(--nav-link-active-shadow); }
            
            .chart-container { height: 300px; }
            .video-shell { background: linear-gradient(135deg, #101827, #0f172a); border-radius: 16px; overflow: hidden; min-height: 320px; position: relative; }
            .video-feed { width: 100%; height: 100%; min-height: 320px; object-fit: cover; display: block; background: #0b1220; }
            .video-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #cbd5e1; font-weight: 600; background: radial-gradient(circle at top, rgba(67,97,238,0.12), rgba(15,23,42,0.98)); }
            .video-meta { position: absolute; left: 16px; bottom: 16px; background: rgba(15,23,42,0.72); color: white; padding: 8px 12px; border-radius: 999px; font-size: 0.85rem; }
            .summary-strip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
            .summary-pill { background: var(--soft-bg); border-radius: 14px; padding: 14px; transition: background-color .25s ease; }
            .summary-label { font-size: 0.78rem; text-transform: uppercase; color: var(--muted-text); font-weight: 700; letter-spacing: .04em; }
            .summary-value { font-size: 1rem; font-weight: 800; margin-top: 6px; }
            .recommend-list { margin: 0; padding-left: 18px; color: var(--muted-text); }
            .recommend-list li { margin-bottom: 8px; }
            .insight-shell { background: var(--soft-bg); border-radius: 16px; padding: 16px; }
            .insight-status { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
            .insight-status.ok { background: rgba(34, 197, 94, 0.12); color: #15803d; }
            .insight-status.watch { background: rgba(245, 158, 11, 0.12); color: #b45309; }
            .insight-status.urgent { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
            .insight-list { margin: 0; padding-left: 18px; color: var(--muted-text); }
            .insight-list li { margin-bottom: 8px; }
            .eval-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
            .eval-card { background: var(--soft-bg); border-radius: 14px; padding: 14px; transition: background-color .25s ease; }
            .eval-card small { display: block; color: var(--muted-text); font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
            .eval-card strong { display: block; font-size: 1.05rem; margin-top: 6px; }
            .node-list { display: grid; gap: 12px; }
            .node-card { background: var(--soft-bg); border-radius: 14px; padding: 14px; border: 1px solid var(--border-soft); transition: background-color .25s ease, border-color .25s ease, opacity .25s ease; }
            .node-card.offline { opacity: 0.68; }
            .node-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
            .node-name { font-weight: 800; }
            .node-meta { color: var(--muted-text); font-size: 0.86rem; }
            .node-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
            .node-pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; background: var(--card-bg); color: var(--text-color); border: 1px solid var(--border-soft); }
            .node-pill.online { background: rgba(34, 197, 94, 0.12); color: #15803d; border-color: rgba(34, 197, 94, 0.2); }
            .node-pill.offline { background: rgba(148, 163, 184, 0.12); color: #64748b; border-color: rgba(148, 163, 184, 0.2); }
            .node-pill.alert { background: rgba(239, 68, 68, 0.12); color: #b91c1c; border-color: rgba(239, 68, 68, 0.2); }
            .node-empty { padding: 28px; text-align: center; color: var(--muted-text); background: var(--soft-bg); border-radius: 14px; }
            .theme-toggle { border: 1px solid var(--badge-border); background: var(--badge-bg); color: var(--badge-text); border-radius: 999px; padding: 8px 14px; font-weight: 600; }
            .theme-toggle:hover { background: var(--soft-bg); color: var(--text-color); }
            .registered-list { display: flex; flex-wrap: wrap; gap: 10px; }
            .registered-pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: var(--soft-bg); border: 1px solid var(--border-soft); font-weight: 700; }
            .ward-pill { background: rgba(67, 97, 238, 0.12); border-color: rgba(67, 97, 238, 0.24); color: var(--primary-color); }
            .registered-remove { border: none; background: transparent; color: #ef4444; font-weight: 800; line-height: 1; padding: 0; }
            .registered-remove:hover { color: #b91c1c; }
            .registration-modal { position: fixed; inset: 0; background: rgba(15, 23, 42, 0.72); display: none; align-items: center; justify-content: center; z-index: 10000; padding: 20px; }
            .registration-modal.show { display: flex; }
            .registration-panel { width: min(760px, 100%); background: var(--card-bg); border-radius: 24px; padding: 24px; box-shadow: 0 24px 60px rgba(0,0,0,0.32); border: 1px solid var(--border-soft); }
            .registration-stage { background: var(--soft-bg); border-radius: 18px; padding: 18px; min-height: 220px; }
            .registration-ref { background: var(--card-bg); border: 1px solid var(--border-soft); border-radius: 18px; min-height: 200px; display: flex; align-items: center; justify-content: center; padding: 12px; }
            .registration-ref svg { width: 180px; height: 180px; }
            .registration-progress { height: 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.18); overflow: hidden; }
            .registration-progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4361ee, #4cc9f0); width: 0%; transition: width .25s ease; }
            .registration-step-pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: rgba(67, 97, 238, 0.12); color: var(--primary-color); font-weight: 700; }
            .text-muted { color: var(--muted-text) !important; }
            .bg-light { background-color: var(--soft-bg) !important; }
            .text-dark { color: var(--badge-text) !important; }
            .border { border-color: var(--badge-border) !important; }
            .list-group-item { background: transparent; color: var(--text-color); border-color: var(--list-border) !important; }
            .card-header { background: var(--card-bg) !important; border-bottom: 1px solid var(--list-border) !important; }
            .input-group .form-control { background: var(--input-bg); color: var(--input-text); border-color: var(--input-border); }
            .input-group .form-control::placeholder { color: var(--muted-text); }
            .btn-group.bg-light { background: var(--soft-bg) !important; }
            @media (max-width: 992px) { .eval-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
            @media (max-width: 576px) { .eval-grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div id="fall-message-display"></div>
        <div id="registration-modal" class="registration-modal">
            <div class="registration-panel">
                <div class="d-flex justify-content-between align-items-start gap-3 mb-3">
                    <div>
                        <h4 class="fw-bold mb-1">Guided Ward Registration</h4>
                        <p id="registration-subtitle" class="text-muted mb-0">We will capture four clear body angles one by one.</p>
                    </div>
                    <button class="btn btn-outline-secondary btn-sm" type="button" onclick="closeRegistrationGuide()">Close</button>
                </div>
                <div class="registration-progress mb-3">
                    <div id="registration-progress-bar" class="registration-progress-bar"></div>
                </div>
                <div class="row g-3">
                    <div class="col-md-7">
                        <div class="registration-stage h-100">
                            <div id="registration-step-pill" class="registration-step-pill mb-3">Waiting</div>
                            <h5 id="registration-title" class="fw-bold mb-2">Waiting for registration</h5>
                            <p id="registration-message" class="text-muted mb-3">Register a visible person to begin the guided capture.</p>
                            <div id="registration-countdown" class="fw-bold fs-5 mb-3"></div>
                            <div id="registration-captured" class="small text-muted mb-3"></div>
                            <button id="registration-capture-btn" class="btn btn-primary px-4" type="button" onclick="captureRegistrationStep()" disabled>Capture This Angle</button>
                        </div>
                    </div>
                    <div class="col-md-5">
                        <div class="registration-ref">
                            <div id="registration-reference"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <nav class="navbar px-4 py-2 sticky-top">
            <div class="container-fluid">
                <a class="navbar-brand fw-bold d-flex align-items-center" href="#">
                    <div class="bg-primary text-white p-2 rounded-3 me-2" style="width: 35px; height: 35px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-heartbeat"></i>
                    </div>
                    ElderlyCare <span class="text-primary ms-1">Pro</span>
                </a>
                <div class="d-flex align-items-center gap-3">
                    <div class="input-group input-group-sm shadow-sm" style="width: 250px;">
                        <input type="text" id="reg-name" class="form-control border-0 px-3" placeholder="Register Name">
                        <button class="btn btn-primary px-3" onclick="registerPerson()">
                            <i class="fas fa-user-plus"></i>
                        </button>
                    </div>
                    <button id="theme-toggle" class="theme-toggle" type="button" onclick="toggleTheme()">
                        <i class="fas fa-moon me-1"></i> Dark Mode
                    </button>
                    <div id="live-time" class="badge bg-light text-dark border p-2 fw-normal"></div>
                </div>
            </div>
        </nav>

        <div class="container-fluid">
            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-2 d-none d-md-block sidebar">
                    <div class="nav flex-column">
                        <a href="/" class="nav-link active"><i class="fas fa-chart-line me-2"></i> Dashboard</a>
                        <a href="#fall-history" class="nav-link"><i class="fas fa-history me-2"></i> Event History</a>
                        <a href="#people-grid" class="nav-link"><i class="fas fa-users me-2"></i> Managed People</a>
                        <a href="/settings" class="nav-link"><i class="fas fa-cog me-2"></i> Settings</a>
                    </div>
                    <hr>
                    <div class="p-3 bg-white rounded-3 shadow-sm mt-4">
                        <div class="small text-muted mb-2">System Status</div>
                        <div class="d-flex align-items-center">
                            <div class="spinner-grow spinner-grow-sm text-success me-2"></div>
                            <span class="small fw-bold">Live Monitoring</span>
                        </div>
                    </div>
                </div>

                <!-- Main Content -->
                <div class="col-md-10 p-4">
                    <div id="alert-container"></div>

                    <div class="row g-4">
                        <!-- Activity Grid -->
                        <div class="col-lg-8">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <h5 class="fw-bold mb-0">Daily Activity Tracker</h5>
                                <div id="unnamed-container" class="d-none">
                                     <div id="unnamed-list" class="d-flex gap-2"></div>
                                </div>
                            </div>

                            <div class="card p-3 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Live Camera Feed</h5>
                                        <p class="text-muted small mb-0">Processed monitoring stream inside the dashboard</p>
                                    </div>
                                </div>
                                <div class="video-shell">
                                    <img id="video-feed" class="video-feed" src="/video_feed" alt="Live monitoring feed">
                                    <div id="video-empty" class="video-empty d-none">Camera feed unavailable</div>
                                    <div class="video-meta" id="video-meta">Connecting to camera...</div>
                                </div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Daily Summary</h5>
                                        <p class="text-muted small mb-0">Saved activity totals and movement recommendations for today</p>
                                    </div>
                                    <div id="summary-date" class="text-muted small"></div>
                                </div>
                                <div id="summary-strip" class="summary-strip mb-4"></div>
                                <div>
                                    <h6 class="fw-bold mb-3">Recommendations</h6>
                                    <ul id="recommendations-list" class="recommend-list"></ul>
                                </div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Caregiver Insights</h5>
                                        <p class="text-muted small mb-0">Live guidance based on alerts, activity balance, and monitoring health</p>
                                    </div>
                                    <div id="caregiver-status" class="insight-status ok">Stable</div>
                                </div>
                                <div class="insight-shell">
                                    <ul id="caregiver-insights-list" class="insight-list"></ul>
                                </div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Registered People</h5>
                                        <p class="text-muted small mb-0">Saved identities available for ward recognition</p>
                                    </div>
                                </div>
                                <div id="registered-people-list" class="registered-list">
                                    <div class="text-muted small">No registered people yet.</div>
                                </div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Evaluation Metrics</h5>
                                        <p class="text-muted small mb-0">Live operational metrics for detection, wake-up, and sync health</p>
                                    </div>
                                    <div id="eval-date" class="text-muted small"></div>
                                </div>
                                <div id="evaluation-grid" class="eval-grid"></div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Video Experiments</h5>
                                        <p class="text-muted small mb-0">Upload recorded activity videos and generate offline pose, activity, and fall analysis results.</p>
                                    </div>
                                </div>
                                <div class="row g-3 align-items-end mb-3">
                                    <div class="col-md-8">
                                        <label class="form-label small text-muted mb-1">Select video</label>
                                        <input id="video-analysis-file" type="file" class="form-control" accept=".mp4,.avi,.mov,.mkv,.m4v">
                                    </div>
                                    <div class="col-md-4 d-grid">
                                        <button class="btn btn-primary" onclick="uploadVideoForAnalysis()">
                                            <i class="fas fa-upload me-2"></i>Upload And Analyze
                                        </button>
                                    </div>
                                </div>
                                <div id="video-analysis-status" class="small text-muted mb-3">No video analysis jobs yet.</div>
                                <div id="video-analysis-jobs" class="row g-3"></div>
                            </div>
                            
                            <div id="people-grid" class="row"></div>

                            <div id="multi-person-banner" class="card p-4 mb-4 d-none">
                                <div class="d-flex align-items-start gap-3">
                                    <div class="bg-warning-subtle text-warning rounded-circle d-flex align-items-center justify-content-center" style="width: 42px; height: 42px;">
                                        <i class="fas fa-users"></i>
                                    </div>
                                    <div>
                                        <h5 class="fw-bold mb-1">Multi-Person Fall Watch</h5>
                                        <p id="multi-person-banner-text" class="text-muted small mb-0">Multiple people are currently around the elderly person. Do not worry. Only fall alerts are active right now, while on-screen activities are shown for reference.</p>
                                    </div>
                                </div>
                            </div>

                            <div id="guest-section" class="card p-4 mb-4 d-none">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Temporary Guests</h5>
                                        <p class="text-muted small mb-0">Live-only guest tracking. This data is not saved to the ward history.</p>
                                    </div>
                                </div>
                                <div id="guest-grid" class="row"></div>
                            </div>

                            <!-- Analytics Section -->
                            <div class="card p-4">
                                <div class="d-flex justify-content-between align-items-center mb-4">
                                    <div>
                                        <h5 class="fw-bold mb-0" id="chart-title">Activity Analytics</h5>
                                        <p class="text-muted small mb-0">Compare walking, sitting, and resting patterns</p>
                                    </div>
                                    <div class="btn-group p-1 bg-light rounded-3">
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-daily" onclick="setViewMode('daily')">Daily</button>
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-monthly" onclick="setViewMode('monthly')">Monthly</button>
                                    </div>
                                </div>
                                <div class="chart-container">
                                    <canvas id="activityChart"></canvas>
                                </div>
                            </div>
                        </div>

                        <!-- Side Panel: Fall Monitor -->
                        <div class="col-lg-4">
                            <h5 class="fw-bold mb-4">Connected Nodes</h5>
                            <div class="card shadow-sm mb-4">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <div>
                                            <div class="fw-bold">Distributed Monitoring</div>
                                            <div class="text-muted small mb-0">Live status for edge laptops connected to this server</div>
                                        </div>
                                        <div id="node-count-pill" class="badge bg-light text-dark border">0 nodes</div>
                                    </div>
                                    <div id="nodes-list" class="node-list">
                                        <div class="node-empty">No remote nodes connected yet.</div>
                                    </div>
                                </div>
                            </div>

                            <h5 class="fw-bold mb-4">Fall Monitor Feed</h5>
                            <div class="card shadow-sm h-100">
                                <div class="card-header bg-white py-3">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span class="small fw-bold text-uppercase text-muted letter-spacing-1">Recent Events</span>
                                        <i class="fas fa-bell text-muted"></i>
                                    </div>
                                </div>
                                <div class="card-body p-0" style="max-height: 700px; overflow-y: auto;">
                                    <div id="fall-history" class="list-group list-group-flush"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let activityChart = null;
            let currentViewMode = 'daily';
            let seenAlerts = new Set();
            let currentTheme = 'light';
            let registrationPollHandle = null;
            let registrationModalOpen = false;

            function applyTheme(theme) {
                currentTheme = theme === 'dark' ? 'dark' : 'light';
                document.body.classList.toggle('theme-dark', currentTheme === 'dark');
                const btn = document.getElementById('theme-toggle');
                if (btn) {
                    btn.innerHTML = currentTheme === 'dark'
                        ? '<i class="fas fa-sun me-1"></i> Light Mode'
                        : '<i class="fas fa-moon me-1"></i> Dark Mode';
                }
                localStorage.setItem('dashboard_theme', currentTheme);
            }

            function toggleTheme() {
                applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
            }

            function loadTheme() {
                const savedTheme = localStorage.getItem('dashboard_theme') || 'light';
                applyTheme(savedTheme);
            }

            function setViewMode(mode) {
                currentViewMode = mode;
                document.getElementById('btn-daily').classList.toggle('bg-white', mode === 'daily');
                document.getElementById('btn-daily').classList.toggle('shadow-sm', mode === 'daily');
                document.getElementById('btn-monthly').classList.toggle('bg-white', mode === 'monthly');
                document.getElementById('btn-monthly').classList.toggle('shadow-sm', mode === 'monthly');
                window.lastChartUpdate = 0;
                update();
            }

            function getAngleReferenceSvg(code) {
                const labels = {
                    front: 'Front',
                    left: 'Left Side',
                    right: 'Right Side',
                    back: 'Back'
                };
                const bodies = {
                    front: '<circle cx="100" cy="56" r="18" fill="#8ecae6"/><rect x="80" y="78" width="40" height="54" rx="18" fill="#4361ee"/><line x1="60" y1="92" x2="140" y2="92" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="90" y1="132" x2="78" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="110" y1="132" x2="122" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/>',
                    left: '<circle cx="100" cy="56" r="18" fill="#8ecae6"/><rect x="88" y="78" width="28" height="54" rx="14" fill="#4361ee"/><line x1="100" y1="92" x2="138" y2="92" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="98" y1="132" x2="92" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="108" y1="132" x2="118" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><polygon points="150,92 170,82 170,102" fill="#ef476f"/>',
                    right: '<circle cx="100" cy="56" r="18" fill="#8ecae6"/><rect x="84" y="78" width="28" height="54" rx="14" fill="#4361ee"/><line x1="62" y1="92" x2="100" y2="92" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="92" y1="132" x2="82" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="102" y1="132" x2="108" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><polygon points="50,92 30,82 30,102" fill="#ef476f"/>',
                    back: '<circle cx="100" cy="56" r="18" fill="#8ecae6"/><rect x="80" y="78" width="40" height="54" rx="18" fill="#4361ee"/><line x1="64" y1="90" x2="136" y2="90" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="92" y1="132" x2="84" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><line x1="108" y1="132" x2="116" y2="174" stroke="#1d3557" stroke-width="10" stroke-linecap="round"/><path d="M82 84 L100 98 L118 84" stroke="#f1f5f9" stroke-width="4" fill="none" stroke-linecap="round"/>'
                };
                return `
                    <svg viewBox="0 0 200 200" role="img" aria-label="Reference pose ${code}">
                        <text x="100" y="26" text-anchor="middle" font-size="18" font-weight="700" fill="#4361ee">${labels[code] || labels.front}</text>
                        ${bodies[code] || bodies.front}
                    </svg>
                `;
            }

            function closeRegistrationGuide() {
                document.getElementById('registration-modal').classList.remove('show');
                registrationModalOpen = false;
                if (registrationPollHandle) {
                    clearTimeout(registrationPollHandle);
                    registrationPollHandle = null;
                }
            }

            function openRegistrationGuide() {
                document.getElementById('registration-modal').classList.add('show');
                registrationModalOpen = true;
            }

            function renderRegistrationStatus(status) {
                const titleEl = document.getElementById('registration-title');
                const msgEl = document.getElementById('registration-message');
                const countdownEl = document.getElementById('registration-countdown');
                const capturedEl = document.getElementById('registration-captured');
                const refEl = document.getElementById('registration-reference');
                const stepPillEl = document.getElementById('registration-step-pill');
                const captureBtn = document.getElementById('registration-capture-btn');
                const progressBar = document.getElementById('registration-progress-bar');
                const subtitleEl = document.getElementById('registration-subtitle');
                captureBtn.textContent = 'Capture This Angle';

                if (!status.active) {
                    captureBtn.disabled = true;
                    if (status.completed) {
                        stepPillEl.textContent = 'Complete';
                        titleEl.textContent = `Registration complete for ${status.name || 'ward'}`;
                        msgEl.textContent = `Automatic 360 capture finished. Saved ${status.captures} body samples for identity persistence.`;
                        countdownEl.textContent = '';
                        capturedEl.textContent = 'You can now close this guide.';
                        progressBar.style.width = '100%';
                    } else {
                        stepPillEl.textContent = 'Waiting';
                        titleEl.textContent = 'Waiting for registration';
                        msgEl.textContent = 'Register a visible person to begin automatic 360 capture.';
                        countdownEl.textContent = '';
                        capturedEl.textContent = '';
                        progressBar.style.width = '0%';
                    }
                    refEl.innerHTML = getAngleReferenceSvg('front');
                    return;
                }

                const progress = Math.max(0, Math.min(100, (status.captures / status.required_captures) * 100));
                progressBar.style.width = `${progress}%`;
                subtitleEl.textContent = `${status.name} is being auto-captured while turning slowly.`;
                capturedEl.textContent = `Saved ${status.captures}/${status.required_captures} target samples.`;

                if (status.state === 'wait_full_body') {
                    stepPillEl.textContent = 'Show Full Body';
                    titleEl.textContent = `Step back, ${status.name}`;
                    msgEl.textContent = 'Registration is queued. As soon as one full body is visible, guided capture will start automatically.';
                    countdownEl.textContent = 'No extra button press is needed.';
                    captureBtn.disabled = true;
                    refEl.innerHTML = getAngleReferenceSvg('front');
                    return;
                }

                if (status.state === 'get_ready') {
                    stepPillEl.textContent = 'Get Ready';
                    titleEl.textContent = `Get ready, ${status.name}`;
                    msgEl.textContent = 'Stand fully in frame. Automatic 360 capture will start after the countdown.';
                    countdownEl.textContent = `Starting in ${status.countdown_sec}s`;
                    captureBtn.disabled = true;
                    refEl.innerHTML = getAngleReferenceSvg('front');
                    return;
                }

                const step = status.step || { code: 'front', label: 'Front View', hint: 'Face the camera directly.' };
                stepPillEl.textContent = 'Auto 360 Capture';
                titleEl.textContent = 'Turn slowly in place';
                msgEl.textContent = 'Keep your full body visible and rotate slowly. The system is saving snapshots automatically.';
                countdownEl.textContent = `${status.countdown_sec}s remaining`;
                captureBtn.disabled = true;
                captureBtn.textContent = 'Auto Capture Running';
                refEl.innerHTML = getAngleReferenceSvg(step.code);
            }

            async function pollRegistrationStatus() {
                try {
                    const res = await fetch('/api/ward-registration/status');
                    const status = await res.json();
                    renderRegistrationStatus(status);
                    if (status.active) {
                        openRegistrationGuide();
                        registrationPollHandle = setTimeout(pollRegistrationStatus, 700);
                    } else if (status.completed) {
                        openRegistrationGuide();
                    } else if (!registrationModalOpen) {
                        closeRegistrationGuide();
                    }
                } catch (err) {
                    console.error(err);
                }
            }

            async function captureRegistrationStep() {
                try {
                    const res = await fetch('/api/ward-registration/capture', { method: 'POST' });
                    const data = await res.json();
                    if (!res.ok || data.status !== 'success') {
                        throw new Error(data.message || 'Capture failed');
                    }
                    renderRegistrationStatus(data.registration || {});
                    loadRegisteredIdentities();
                    if (data.registration && data.registration.active) {
                        registrationPollHandle = setTimeout(pollRegistrationStatus, 300);
                    }
                } catch (err) {
                    console.error(err);
                    alert(err.message || 'Could not capture this angle.');
                    registrationPollHandle = setTimeout(pollRegistrationStatus, 300);
                }
            }

            async function loadRegisteredIdentities() {
                try {
                    const res = await fetch('/api/registered-identities');
                    const data = await res.json();
                    const container = document.getElementById('registered-people-list');
                    if (!data.names || !data.names.length) {
                        container.innerHTML = '<div class="text-muted small">No registered people yet.</div>';
                        return;
                    }
                    container.innerHTML = data.names.map(name => `
                        <div class="registered-pill ${name === data.ward_name ? 'ward-pill' : ''}">
                            <span>${name}${name === data.ward_name ? ' (Ward)' : ''}</span>
                            <button class="registered-remove" type="button" onclick="deleteRegisteredIdentity('${name.replace(/'/g, "\\'")}')" title="Remove registered person">&times;</button>
                        </div>
                    `).join('');
                } catch (err) {
                    console.error(err);
                }
            }

            async function deleteRegisteredIdentity(name) {
                if (!confirm(`Remove registered person "${name}"?`)) return;
                try {
                    const res = await fetch(`/api/registered-identities/${encodeURIComponent(name)}`, { method: 'DELETE' });
                    const data = await res.json();
                    if (!res.ok || data.status !== 'success') {
                        throw new Error(data.message || 'Delete failed');
                    }
                    loadRegisteredIdentities();
                } catch (err) {
                    console.error(err);
                    alert(err.message || 'Could not remove registered person.');
                }
            }

            function registerPerson(yoloId = null) {
                const name = document.getElementById('reg-name').value;
                if (!name) { alert('Enter a name first'); return; }
                const formData = new FormData();
                formData.append('name', name);
                if (yoloId) formData.append('yolo_id', yoloId);
                fetch('/register', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        if(data.status === 'success') {
                            document.getElementById('reg-name').value = '';
                            if (data.guided_registration) {
                                openRegistrationGuide();
                                pollRegistrationStatus();
                                loadRegisteredIdentities();
                            }
                        }
                        alert(data.message);
                    });
            }

            function initChart(data) {
                const ctx = document.getElementById('activityChart').getContext('2d');
                if (activityChart) activityChart.destroy();
                
                const labels = [...new Set(data.map(d => d.date))].reverse();
                const datasets = [];
                const colors = { 
                    walk: '#10b981', 
                    stand: '#3b82f6', 
                    sit: '#f59e0b', 
                    sleep: '#8b5cf6' 
                };

                ['walk', 'stand', 'sit', 'sleep'].forEach(type => {
                    datasets.push({
                        label: type.charAt(0).toUpperCase() + type.slice(1),
                        data: labels.map(l => {
                            const entries = data.filter(d => d.date === l);
                            const val = entries.reduce((acc, curr) => acc + curr[type], 0);
                            // Convert to hours for monthly, minutes for daily
                            return currentViewMode === 'monthly' ? (val / 3600).toFixed(1) : (val / 60).toFixed(1);
                        }),
                        backgroundColor: colors[type],
                        borderColor: colors[type],
                        borderWidth: 1,
                        borderRadius: 4
                    });
                });

                activityChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels, datasets },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        plugins: { 
                            legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 6 } },
                            tooltip: { backgroundColor: '#1e1e2f', padding: 12 }
                        },
                        scales: { 
                            y: { 
                                beginAtZero: true, 
                                title: { display: true, text: currentViewMode === 'monthly' ? 'Hours' : 'Minutes' },
                                grid: { color: '#f0f0f0' }
                            },
                            x: { grid: { display: false } }
                        } 
                    }
                });
            }

            function showFallToast(msg, type) {
                const container = document.getElementById('fall-message-display');
                const toast = document.createElement('div');
                toast.className = 'toast-fall';
                if (type === 'RECOVERED') {
                    toast.style.background = '#10b981';
                    toast.innerHTML = `<i class="fas fa-check-circle me-3"></i> <div><strong>RECOVERY</strong><br><small>${msg}</small></div>`;
                } else {
                    toast.innerHTML = `<i class="fas fa-exclamation-triangle me-3"></i> <div><strong>FALL ALERT</strong><br><small>${msg}</small></div>`;
                }
                container.appendChild(toast);
                setTimeout(() => {
                    toast.style.opacity = '0';
                    setTimeout(() => toast.remove(), 300);
                }, 6000);
            }

            function ackFall(pid) {
                fetch(`/api/acknowledge/${pid}`, { method: 'POST' }).then(() => update());
            }

            function renderDailySummary(summary) {
                document.getElementById('summary-date').textContent = summary.date || '';
                const items = [
                    ['Walking', summary.walking_dur],
                    ['Standing', summary.standing_dur],
                    ['Sitting', summary.sitting_dur],
                    ['Sleeping', summary.sleeping_dur],
                    ['Monitored', summary.monitored_dur]
                ];
                document.getElementById('summary-strip').innerHTML = items.map(([label, value]) => `
                    <div class="summary-pill">
                        <div class="summary-label">${label}</div>
                        <div class="summary-value">${value}</div>
                    </div>
                `).join('');

                document.getElementById('recommendations-list').innerHTML = (summary.recommendations || []).map(item => `
                    <li>${item}</li>
                `).join('');
            }

            function renderCaregiverInsights(summary) {
                const statusEl = document.getElementById('caregiver-status');
                const insights = summary.caregiver_insights || [];
                const status = summary.caregiver_status || 'ok';
                const labels = { ok: 'Stable', watch: 'Watch', urgent: 'Urgent' };
                statusEl.className = `insight-status ${status}`;
                statusEl.textContent = labels[status] || 'Stable';
                document.getElementById('caregiver-insights-list').innerHTML = insights.map(item => `
                    <li>${item}</li>
                `).join('') || '<li>No caregiver insights yet.</li>';
            }

            function renderEvaluation(metrics) {
                document.getElementById('eval-date').textContent = metrics.date || '';
                const items = [
                    ['FPS', metrics.fps ?? 0],
                    ['People Tracked', metrics.people_tracked ?? 0],
                    ['Falls Today', metrics.falls_today ?? 0],
                    ['Major Falls Today', metrics.major_falls_today ?? 0],
                    ['Low Power Entries', metrics.low_power_entries ?? 0],
                    ['Wake Events', metrics.wake_events ?? 0],
                    ['Last Wake (s)', metrics.last_wake_latency_sec ?? 'N/A'],
                    ['Avg Wake (s)', metrics.avg_wake_latency_sec ?? 'N/A'],
                    ['Sync OK', metrics.central_sync_ok ?? 0],
                    ['Sync Fail', metrics.central_sync_fail ?? 0],
                    ['Connected Nodes', metrics.connected_nodes ?? 0],
                    ['Has Frame', metrics.has_frame ? 'Yes' : 'No']
                ];
                document.getElementById('evaluation-grid').innerHTML = items.map(([label, value]) => `
                    <div class="eval-card">
                        <small>${label}</small>
                        <strong>${value}</strong>
                    </div>
                `).join('');
            }

            function renderNodes(nodes) {
                const list = document.getElementById('nodes-list');
                const countPill = document.getElementById('node-count-pill');
                countPill.textContent = `${nodes.length} node${nodes.length === 1 ? '' : 's'}`;

                if (!nodes.length) {
                    list.innerHTML = '<div class="node-empty">No remote nodes connected yet.</div>';
                    return;
                }

                list.innerHTML = nodes.map(node => {
                    const statusClass = node.is_online ? 'online' : 'offline';
                    const cameraLabel = node.camera_available ? 'Camera ready' : 'No camera';
                    const frameLabel = node.has_frame ? 'Live frames' : 'No frames';
                    const peopleLabel = `${node.people_count || 0} people`;
                    const alertCount = node.active_alert_count || 0;
                    const alertLabel = `${alertCount} alert${alertCount === 1 ? '' : 's'}`;
                    const seenLabel = `${node.seconds_since_seen ?? 'N/A'}s ago`;
                    const reportLabel = `${node.last_report_age_sec ?? 'N/A'}s ago`;
                    return `
                        <div class="node-card ${node.is_online ? '' : 'offline'}">
                            <div class="node-head">
                                <div>
                                    <div class="node-name">${node.node_id}</div>
                                    <div class="node-meta">${node.address || 'Unknown address'} • ${node.deployment_mode || 'edge'}</div>
                                </div>
                                <span class="node-pill ${statusClass}">
                                    <i class="fas ${node.is_online ? 'fa-circle-check' : 'fa-circle-minus'}"></i>
                                    ${node.is_online ? 'Online' : 'Offline'}
                                </span>
                            </div>
                            <div class="node-meta">Last seen ${seenLabel} • Last report ${reportLabel}</div>
                            <div class="node-pills">
                                <span class="node-pill">${cameraLabel}</span>
                                <span class="node-pill">${frameLabel}</span>
                                <span class="node-pill">${peopleLabel}</span>
                                <span class="node-pill ${alertCount > 0 ? 'alert' : ''}">${alertLabel}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            }

            function renderPersonCards(targetId, people, options = {}) {
                const container = document.getElementById(targetId);
                const multiPersonNotice = options.multiPersonMode && targetId === 'people-grid'
                    ? `
                        <div class="col-12 mb-4">
                            <div class="card p-4 border-0 shadow-sm" style="background: linear-gradient(135deg, rgba(255,248,220,0.98), rgba(255,255,255,0.98));">
                                <div class="d-flex align-items-start">
                                    <div class="me-3 mt-1 text-warning"><i class="fas fa-user-friends fa-lg"></i></div>
                                    <div>
                                        <h6 class="fw-bold mb-1">Multiple People In Frame</h6>
                                        <p class="mb-0 text-muted small">${options.visiblePeopleCount || 0} people are currently around the elderly person. Do not worry. Only fall alerts are active right now, while the activities below are shown for reference.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `
                    : '';
                const peopleHtml = people.map(p => {
                    let activity = p.current_activity;
                    let badgeCls = ['WALKING', 'STANDING', 'SITTING', 'SLEEPING', 'LYING', 'TRANSITION', 'MAJOR FALL', 'MINOR FALL', 'RECOVERED'].includes(activity)
                        ? 'bg-' + activity.toLowerCase().replace(' ', '-')
                        : 'bg-away';
                    const seenFor = p.seen_for_dur ? `<div class="small text-muted mt-2">Seen for ${p.seen_for_dur}</div>` : '';
                    return `
                        <div class="col-md-6 mb-4" style="opacity: ${p.is_active ? '1.0' : '0.6'}">
                            <div class="card p-4 h-100">
                                <div class="d-flex justify-content-between align-items-center mb-4">
                                    <div class="d-flex align-items-center">
                                        <div class="bg-light p-2 rounded-circle me-3" style="width: 45px; height: 45px; display: flex; align-items: center; justify-content: center;">
                                            <i class="fas fa-user text-muted"></i>
                                        </div>
                                        <div>
                                            <h6 class="fw-bold mb-0">${p.person}</h6>
                                            ${p.role === 'guest' ? '<div class="small text-muted">Temporary guest</div>' : ''}
                                        </div>
                                    </div>
                                    <span class="status-badge ${badgeCls}">${activity}</span>
                                </div>
                                <div class="row g-2">
                                    <div class="col-3 text-center">
                                        <div class="stat-icon icon-walk mx-auto"><i class="fas fa-walking"></i></div>
                                        <div class="fw-bold small">${p.walking_dur}</div>
                                    </div>
                                    <div class="col-3 text-center">
                                        <div class="stat-icon icon-stand mx-auto"><i class="fas fa-male"></i></div>
                                        <div class="fw-bold small">${p.standing_dur}</div>
                                    </div>
                                    <div class="col-3 text-center">
                                        <div class="stat-icon icon-sit mx-auto"><i class="fas fa-chair"></i></div>
                                        <div class="fw-bold small">${p.sitting_dur}</div>
                                    </div>
                                    <div class="col-3 text-center">
                                        <div class="stat-icon icon-sleep mx-auto"><i class="fas fa-bed"></i></div>
                                        <div class="fw-bold small">${p.sleeping_dur}</div>
                                    </div>
                                </div>
                                ${seenFor}
                            </div>
                        </div>`;
                }).join('');
                container.innerHTML = multiPersonNotice + (peopleHtml || '<div class="col-12"><div class="card p-5 text-center text-muted">No live detections.</div></div>');
            }

            function renderVideoAnalysisJobs(payload) {
                const jobs = payload.jobs || [];
                const status = document.getElementById('video-analysis-status');
                const container = document.getElementById('video-analysis-jobs');
                status.textContent = payload.running_count > 0
                    ? `${payload.running_count} analysis job(s) currently running.`
                    : (jobs.length ? 'Latest uploaded experiment videos are shown below.' : 'No video analysis jobs yet.');

                container.innerHTML = jobs.map(job => {
                    const summary = job.summary || {};
                    const events = summary.events || [];
                    const graphUrls = job.graph_urls || {};
                    const eventHtml = events.slice(0, 5).map(evt => `
                        <div class="small text-muted">${evt.time_label} - ${evt.event}</div>
                    `).join('');
                    const outputLink = job.output_video_url
                        ? `<div class="d-flex gap-2 flex-wrap mt-2">
                                <a class="btn btn-sm btn-primary" href="${job.viewer_url}" target="_blank">View Annotated Video</a>
                                <a class="btn btn-sm btn-outline-primary" href="${job.output_video_url}" target="_blank">Open Raw Saved File</a>
                           </div>`
                        : '';
                    return `
                        <div class="col-12">
                            <div class="card p-4 h-100">
                                <div class="d-flex justify-content-between align-items-start gap-3 mb-3">
                                    <div>
                                        <div class="fw-bold">${job.original_name || job.input_name || 'Uploaded video'}</div>
                                        <div class="small text-muted">Status: ${job.status} • Updated ${job.updated_at_label || ''}</div>
                                    </div>
                                    <span class="status-badge ${job.status === 'completed' ? 'bg-standing' : (job.status === 'failed' ? 'bg-major-fall' : 'bg-sitting')}">${job.status.toUpperCase()}</span>
                                </div>
                                <div class="row g-2 mb-2">
                                    <div class="col-md-3"><div class="small text-muted">Frames</div><div class="fw-bold">${summary.frames_processed ?? '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Proc. FPS</div><div class="fw-bold">${summary.processing_fps ?? '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Minor Falls</div><div class="fw-bold">${summary.minor_falls ?? 0}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Major Falls</div><div class="fw-bold">${summary.major_falls ?? 0}</div></div>
                                </div>
                                <div class="row g-2">
                                    <div class="col-md-3"><div class="small text-muted">Walking</div><div class="fw-bold">${summary.walking_dur || '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Standing</div><div class="fw-bold">${summary.standing_dur || '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Sitting</div><div class="fw-bold">${summary.sitting_dur || '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Sleeping</div><div class="fw-bold">${summary.sleeping_dur || '-'}</div></div>
                                </div>
                                <div class="row g-2 mt-1">
                                    <div class="col-md-3"><div class="small text-muted">Frame Time (ms)</div><div class="fw-bold">${summary.avg_frame_time_ms ?? '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Avg Pose Conf</div><div class="fw-bold">${summary.avg_pose_confidence ?? '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Pose Coverage (%)</div><div class="fw-bold">${summary.pose_coverage_pct ?? '-'}</div></div>
                                    <div class="col-md-3"><div class="small text-muted">Body Coverage (%)</div><div class="fw-bold">${summary.body_coverage_pct ?? '-'}</div></div>
                                </div>
                                <div class="row g-2 mt-1">
                                    <div class="col-md-4"><div class="small text-muted">Multi-person Coverage (%)</div><div class="fw-bold">${summary.multi_person_detection_coverage_pct ?? '-'}</div></div>
                                    <div class="col-md-4"><div class="small text-muted">Fall Detection Rate (%)</div><div class="fw-bold">${summary.fall_detection_rate_pct ?? 'N/A'}</div></div>
                                    <div class="col-md-4"><div class="small text-muted">False Alerts</div><div class="fw-bold">${summary.false_alerts ?? 'N/A'}</div></div>
                                </div>
                                ${(graphUrls.activity_distribution || graphUrls.fall_events) ? `
                                    <div class="row g-3 mt-2">
                                        ${graphUrls.activity_distribution ? `
                                            <div class="col-lg-6">
                                                <div class="small fw-bold mb-2">Activity Duration Graph</div>
                                                <img src="${graphUrls.activity_distribution}" alt="Activity graph" class="img-fluid rounded border">
                                            </div>` : ''}
                                        ${graphUrls.fall_events ? `
                                            <div class="col-lg-6">
                                                <div class="small fw-bold mb-2">Fall/Event Graph</div>
                                                <img src="${graphUrls.fall_events}" alt="Fall graph" class="img-fluid rounded border">
                                            </div>` : ''}
                                    </div>
                                ` : ''}
                                ${job.error ? `<div class="small text-danger mt-3">${job.error}</div>` : ''}
                                ${eventHtml ? `<div class="mt-3"><div class="fw-bold small mb-1">Detected events</div>${eventHtml}</div>` : ''}
                                ${outputLink}
                            </div>
                        </div>
                    `;
                }).join('');
            }

            async function refreshVideoAnalysisJobs() {
                try {
                    const res = await fetch('/api/video-analysis/jobs');
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Failed to load analysis jobs');
                    renderVideoAnalysisJobs(data);
                } catch (err) {
                    console.error(err);
                    document.getElementById('video-analysis-status').textContent = `Could not load jobs: ${err.message}`;
                }
            }

            async function uploadVideoForAnalysis() {
                const fileInput = document.getElementById('video-analysis-file');
                if (!fileInput.files || !fileInput.files.length) {
                    alert('Please choose a video file first.');
                    return;
                }
                const formData = new FormData();
                formData.append('video', fileInput.files[0]);
                document.getElementById('video-analysis-status').textContent = 'Uploading video for analysis...';
                try {
                    const res = await fetch('/api/video-analysis/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Upload failed');
                    fileInput.value = '';
                    document.getElementById('video-analysis-status').textContent = data.message || 'Video uploaded successfully.';
                    refreshVideoAnalysisJobs();
                } catch (err) {
                    console.error(err);
                    document.getElementById('video-analysis-status').textContent = `Upload failed: ${err.message}`;
                    alert(`Upload failed: ${err.message}`);
                }
            }

            function update() {
                fetch('/api/report')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('live-time').innerHTML = `<i class="far fa-clock me-1 text-muted"></i> ${new Date().toLocaleTimeString()}`;
                        const multiBanner = document.getElementById('multi-person-banner');
                        const multiBannerText = document.getElementById('multi-person-banner-text');
                        const guestSection = document.getElementById('guest-section');
                        if (data.multi_person_mode) {
                            multiBanner.classList.remove('d-none');
                            const peopleCount = data.visible_people_count || 0;
                            multiBannerText.textContent = `${peopleCount} people are currently around the elderly person. Do not worry. Only fall alerts are active right now, while each person's on-screen activity is shown for reference.`;
                            guestSection.classList.add('d-none');
                            document.getElementById('guest-grid').innerHTML = '';
                        } else {
                            multiBanner.classList.add('d-none');
                        }
                        
                        // Active Alerts
                        let alertHtml = '';
                        for (let l of data.active_alerts) {
                            const isRecovered = l.type === 'RECOVERED';
                            const alertKey = `${l.person_id}-${l.type}-${l.timestamp}`;
                            if (!seenAlerts.has(alertKey)) {
                                showFallToast(`${l.message}`, l.type);
                                seenAlerts.add(alertKey);
                            }
                            
                            alertHtml += `
                                <div class="alert-item shadow-sm ${isRecovered ? 'recovered' : ''}">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h6 class="mb-1 ${isRecovered ? 'text-success' : 'text-danger'} fw-bold">
                                                ${isRecovered ? '<i class="fas fa-check-circle"></i> RECOVERY' : '<i class="fas fa-exclamation-circle"></i> FALL ALERT'}
                                            </h6>
                                            <p class="mb-0 small text-muted"><strong>${l.message}</strong> - Detected at ${l.time_str}</p>
                                        </div>
                                        <button class="btn btn-outline-dark btn-sm rounded-pill px-3" onclick="ackFall('${l.person_id}')">Dismiss</button>
                                    </div>
                                </div>`;
                        }
                        document.getElementById('alert-container').innerHTML = alertHtml;
                        
                        // Unnamed IDs
                        const unnamedContainer = document.getElementById('unnamed-container');
                        if (data.unnamed_ids && data.unnamed_ids.length > 0) {
                            unnamedContainer.classList.remove('d-none');
                            document.getElementById('unnamed-list').innerHTML = data.unnamed_ids.map(id => 
                                `<button class="btn btn-warning btn-sm border-0 rounded-2 px-3 shadow-sm" onclick="registerPerson('${id}')">Name ID ${id}</button>`
                            ).join('');
                        } else {
                            unnamedContainer.classList.add('d-none');
                        }

                        renderPersonCards('people-grid', data.people || [], {
                            multiPersonMode: data.multi_person_mode,
                            visiblePeopleCount: data.visible_people_count
                        });
                        if (!data.multi_person_mode && data.guests && data.guests.length > 0) {
                            guestSection.classList.remove('d-none');
                            renderPersonCards('guest-grid', data.guests);
                        } else {
                            guestSection.classList.add('d-none');
                            document.getElementById('guest-grid').innerHTML = '';
                        }
                        
                        // Fall History
                        document.getElementById('fall-history').innerHTML = data.falls.map(f => {
                            let icon = f.type === 'MAJOR FALL' ? 'fa-exclamation-circle text-danger' : 
                                      (f.type === 'RECOVERED' ? 'fa-check-circle text-success' : 'fa-exclamation-triangle text-warning');
                            return `
                                <div class="list-group-item px-4 py-3 border-0 border-bottom">
                                    <div class="d-flex align-items-center">
                                        <i class="fas ${icon} fs-5 me-3"></i>
                                        <div class="flex-grow-1">
                                            <div class="fw-bold small">${f.person}</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">${f.type} • ${f.time_str}</div>
                                        </div>
                                    </div>
                                </div>`;
                        }).join('') || '<div class="p-5 text-center text-muted small">No fall history yet.</div>';
                    });

                fetch('/api/daily-summary')
                    .then(r => r.json())
                    .then(summary => {
                        renderDailySummary(summary);
                        renderCaregiverInsights(summary);
                    });

                fetch('/api/evaluation')
                    .then(r => r.json())
                    .then(metrics => renderEvaluation(metrics));

                fetch('/api/nodes')
                    .then(r => r.json())
                    .then(nodes => renderNodes(nodes));

                fetch('/api/camera-status')
                    .then(r => r.json())
                    .then(status => {
                        const empty = document.getElementById('video-empty');
                        const meta = document.getElementById('video-meta');
                        if (status.has_frame) {
                            empty.classList.add('d-none');
                            meta.textContent = 'Live feed active';
                        } else if (!status.camera_available) {
                            empty.classList.remove('d-none');
                            meta.textContent = 'No camera detected';
                        } else {
                            empty.classList.remove('d-none');
                            meta.textContent = 'Waiting for frames...';
                        }
                    });

                // Update charts
                if (!window.lastChartUpdate || (Date.now() - window.lastChartUpdate > 30000)) {
                    const endpoint = currentViewMode === 'daily' ? '/api/history' : '/api/history/monthly';
                    fetch(endpoint).then(r => r.json()).then(data => {
                        initChart(data);
                        window.lastChartUpdate = Date.now();
                    });
                }
            }
            setInterval(update, 2000);
            loadTheme();
            loadRegisteredIdentities();
            pollRegistrationStatus();
            refreshVideoAnalysisJobs();
            update();
            setViewMode('daily');
            setInterval(refreshVideoAnalysisJobs, 4000);
        </script>
    </body>
    </html>
    """
    return html

def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def build_recommendations(walk_s, stand_s, sit_s, sleep_s, monitored_s):
    recommendations = []
    if monitored_s <= 0:
        return ["Not enough activity data yet. Keep the camera on during the day to build recommendations."]

    if walk_s < 30 * 60:
        recommendations.append("Walk more today. Aim for at least 30 minutes of walking spread through the day.")
    if sit_s > 2 * 60 * 60:
        recommendations.append("Do not sit too long. Try standing up or stretching every 30 to 60 minutes.")
    if sit_s > walk_s * 3:
        recommendations.append("Movement is low compared with sitting time. Add short walking breaks more often.")
    if stand_s < 45 * 60:
        recommendations.append("Spend a bit more time standing or moving around instead of staying seated.")
    if sleep_s > monitored_s * 0.6 and monitored_s > 2 * 60 * 60:
        recommendations.append("A large part of the monitored time was spent lying down. Check if daytime rest is becoming excessive.")

    if not recommendations:
        recommendations.append("Today's activity balance looks healthy. Keep following the same routine.")
    return recommendations

def build_caregiver_insights(walk_s, stand_s, sit_s, sleep_s, monitored_s):
    with data_lock:
        active_alert_snapshot = list(active_alerts)
        recent_falls = list(fall_events[-5:])
        active_states = list(person_state.values())
        connected_node_count = len(remote_nodes)
        offline_node_count = sum(1 for node in remote_nodes.values() if (time.time() - node.get("last_seen", 0)) > 10.0)

    insights = []
    status = "ok"

    if active_alert_snapshot:
        insights.append(f"{len(active_alert_snapshot)} active alert(s) need caregiver attention right now.")
        status = "urgent"

    if any(event.get("type") == "MAJOR FALL" for event in recent_falls):
        insights.append("A major fall was recorded recently. Verify recovery and reassess the environment for hazards.")
        status = "urgent"

    if monitored_s <= 0:
        insights.append("No meaningful activity data has been collected yet. Keep the resident in frame to generate guidance.")
        return status, insights

    if walk_s < 20 * 60:
        insights.append("Walking time is low today. Encourage a short supervised walk if appropriate.")
        status = "watch" if status == "ok" else status

    if sit_s > 2 * 60 * 60:
        insights.append("Sitting time is prolonged. A posture change or brief standing break would be helpful.")
        status = "watch" if status == "ok" else status

    if sleep_s > monitored_s * 0.55 and monitored_s > 2 * 60 * 60:
        insights.append("Lying time is dominating the monitored period. Check whether daytime rest is higher than usual.")
        status = "watch" if status == "ok" else status

    if stand_s < 20 * 60 and walk_s < 20 * 60:
        insights.append("Overall mobility is limited so far. Monitor for fatigue, pain, or low confidence when moving.")
        status = "watch" if status == "ok" else status

    if active_states and all(state in ["SLEEPING", "LYING"] for state in active_states):
        insights.append("The resident is currently down or resting. Confirm that the posture is intentional and comfortable.")
        status = "watch" if status == "ok" else status

    if connected_node_count > 0:
        if offline_node_count > 0:
            insights.append(f"{offline_node_count} remote node(s) appear offline. Check network or power on those laptops.")
            status = "watch" if status == "ok" else status
        else:
            insights.append(f"All {connected_node_count} connected monitoring node(s) are reporting normally.")

    if not insights:
        insights.append("Activity and monitoring signals look stable right now. Continue routine observation.")

    return status, insights

def get_daily_summary():
    with data_lock:
        walk_s = float(walking_time.get(SINGLE_PERSON_LABEL, 0))
        stand_s = float(standing_time.get(SINGLE_PERSON_LABEL, 0))
        sit_s = float(sitting_time.get(SINGLE_PERSON_LABEL, 0))
        sleep_s = float(sleeping_time.get(SINGLE_PERSON_LABEL, 0))

    monitored_s = walk_s + stand_s + sit_s + sleep_s
    recommendations = build_recommendations(walk_s, stand_s, sit_s, sleep_s, monitored_s)
    caregiver_status, caregiver_insights = build_caregiver_insights(walk_s, stand_s, sit_s, sleep_s, monitored_s)
    return {
        "date": str(date.today()),
        "walking_seconds": walk_s,
        "standing_seconds": stand_s,
        "sitting_seconds": sit_s,
        "sleeping_seconds": sleep_s,
        "monitored_seconds": monitored_s,
        "walking_dur": format_duration(walk_s),
        "standing_dur": format_duration(stand_s),
        "sitting_dur": format_duration(sit_s),
        "sleeping_dur": format_duration(sleep_s),
        "monitored_dur": format_duration(monitored_s),
        "recommendations": recommendations,
        "caregiver_status": caregiver_status,
        "caregiver_insights": caregiver_insights
    }

@app.route("/api/daily-summary")
def api_daily_summary():
    return jsonify(get_daily_summary())

@app.route("/api/evaluation")
def api_evaluation():
    return jsonify(get_evaluation_snapshot())

def build_report_snapshot(include_remote=True):
    with data_lock:
        # Snapshot current state for reporting
        if SINGLE_PERSON_MODE:
            current_state_snapshot = {}
            active_states = [state for state in person_state.values() if state != "UNKNOWN"]
            if active_states:
                current_state_snapshot[SINGLE_PERSON_LABEL] = active_states[0]
        else:
            current_state_snapshot = person_state.copy()
        # Unnamed IDs are those in current state that don't have a manual mapping
        unnamed_ids = [] if SINGLE_PERSON_MODE else [pid for pid in current_state_snapshot if pid not in manual_id_map]
        primary_id = get_primary_monitored_id()
        multi_person_mode = bool(multi_person_scene_active)
        visible_people_count_snapshot = int(multi_person_scene_count)
        
        # Sort people: currently active first, then by total monitored time
        all_pids = list(all_tracked_people)
        
        def get_activity_score(display_id):
            # Check if this person (by name or ID) is currently active
            is_active = display_id in current_state_snapshot # If it's a persistent_id
            if not is_active:
                # Check if any persistent_id mapped to this name is active
                is_active = any(k in current_state_snapshot for k, v in manual_id_map.items() if v == display_id)
            
            total_time = walking_time.get(display_id, 0) + standing_time.get(display_id, 0) + sitting_time.get(display_id, 0) + sleeping_time.get(display_id, 0)
            return (is_active, total_time)

        sorted_pids = sorted(all_pids, key=get_activity_score, reverse=True)
        
        report_data = []
        for display_id in sorted_pids[:10]:
            # Try to find an ACTIVE persistent ID for this name to get the current activity
            internal_id = display_id
            if not SINGLE_PERSON_MODE:
                for k, v in manual_id_map.items():
                    if v == display_id:
                        internal_id = k
                        if k in current_state_snapshot:
                            break # Prioritize the active one

            current_activity = current_state_snapshot.get(internal_id, "AWAY")

            report_data.append({
                "person": display_id,
                "walking_dur": format_duration(walking_time.get(display_id, 0)),
                "standing_dur": format_duration(standing_time.get(display_id, 0)),
                "sleeping_dur": format_duration(sleeping_time.get(display_id, 0)),
                "sitting_dur": format_duration(sitting_time.get(display_id, 0)),
                "current_activity": current_activity,
                "is_active": internal_id in current_state_snapshot,
                "role": "ward" if str(internal_id) == str(primary_id) or display_id == SINGLE_PERSON_LABEL else "resident"
            })

        guest_report = []
        if not multi_person_mode:
            guest_ids = sorted(
                guest_state.keys(),
                key=lambda pid: guest_last_seen.get(pid, 0),
                reverse=True
            )
            for guest_id in guest_ids[:6]:
                guest_report.append({
                    "person": get_guest_display_name(guest_id),
                    "walking_dur": format_duration(guest_walking_time.get(guest_id, 0)),
                    "standing_dur": format_duration(guest_standing_time.get(guest_id, 0)),
                    "sleeping_dur": format_duration(guest_sleeping_time.get(guest_id, 0)),
                    "sitting_dur": format_duration(guest_sitting_time.get(guest_id, 0)),
                    "current_activity": guest_state.get(guest_id, "AWAY"),
                    "is_active": (time.time() - guest_last_seen.get(guest_id, 0)) <= 5.0,
                    "role": "guest",
                    "seen_for_dur": format_duration(max(0.0, guest_last_seen.get(guest_id, 0) - guest_first_seen.get(guest_id, guest_last_seen.get(guest_id, 0))))
                })
            
        # Ensure alerts and history are sorted by high-precision float timestamp
        alerts_copy = sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)
        falls_copy = sorted(fall_events, key=lambda x: x['timestamp'], reverse=True)[:10]

    snapshot = {
        "people": report_data,
        "guests": guest_report,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids,
        "multi_person_mode": multi_person_mode,
        "visible_people_count": visible_people_count_snapshot
    }

    if include_remote:
        with data_lock:
            remote_reports = list(remote_edge_reports.values())
        remote_reports.sort(key=lambda item: item.get("updated_at", 0), reverse=True)

        for remote in remote_reports:
            node_id = remote.get("node_id", "remote")
            for person in remote.get("people", []):
                merged_person = dict(person)
                merged_person["person"] = f"{merged_person.get('person', 'Unknown')} [{node_id}]"
                snapshot["people"].append(merged_person)
            for guest in remote.get("guests", []):
                merged_guest = dict(guest)
                merged_guest["person"] = f"{merged_guest.get('person', 'Unknown')} [{node_id}]"
                snapshot["guests"].append(merged_guest)
            for alert in remote.get("active_alerts", []):
                merged_alert = dict(alert)
                merged_alert["person_id"] = f"{merged_alert.get('person_id', 'Unknown')} [{node_id}]"
                merged_alert["message"] = f"[{node_id}] {merged_alert.get('message', '')}"
                snapshot["active_alerts"].append(merged_alert)
            for fall_item in remote.get("falls", []):
                merged_fall = dict(fall_item)
                merged_fall["person"] = f"{merged_fall.get('person', 'Unknown')} [{node_id}]"
                snapshot["falls"].append(merged_fall)

        snapshot["active_alerts"] = sorted(snapshot["active_alerts"], key=lambda x: x['timestamp'], reverse=True)
        snapshot["falls"] = sorted(snapshot["falls"], key=lambda x: x['timestamp'], reverse=True)[:10]

    return snapshot

@app.route("/api/report")
def api_report():
    """API endpoint for JSON report with sorting and limiting"""
    return jsonify(build_report_snapshot(include_remote=True))

@app.route("/api/ward-registration/status")
def api_ward_registration_status():
    return jsonify(get_ward_registration_status())

@app.route("/api/ward-registration/capture", methods=["POST"])
def api_ward_registration_capture():
    ok, message, step = perform_ward_registration_capture()
    payload = {
        "status": "success" if ok else "error",
        "message": message,
        "step": step,
        "registration": get_ward_registration_status()
    }
    status_code = 200 if ok else 400
    return jsonify(payload), status_code

@app.route("/api/registered-identities")
def api_registered_identities():
    return jsonify({
        "names": get_registered_identities(),
        "ward_name": ward_profile.get("name", "")
    })

@app.route("/api/registered-identities/<name>", methods=["DELETE"])
def api_delete_registered_identity(name):
    ok, message = delete_registered_identity(name)
    status_code = 200 if ok else 404
    return jsonify({
        "status": "success" if ok else "error",
        "message": message,
        "names": get_registered_identities(),
        "ward_name": ward_profile.get("name", "")
    }), status_code

def serialize_video_analysis_job(job):
    result = dict(job)
    updated_at = float(result.get("updated_at", 0) or 0)
    result["updated_at_label"] = time.strftime("%H:%M:%S", time.localtime(updated_at)) if updated_at else ""
    output_video = result.get("output_video")
    result["output_video_url"] = f"/video-analysis/results/{os.path.basename(output_video)}" if output_video else ""
    result["stream_url"] = f"/video-analysis/stream/{result.get('id', '')}" if output_video else ""
    result["viewer_url"] = f"/video-analysis/view/{result.get('id', '')}" if output_video else ""
    graph_files = result.get("graph_files") or {}
    cache_tag = int(updated_at) if updated_at else 0
    result["graph_urls"] = {
        key: f"/video-analysis/graphs/{os.path.basename(path)}?v={cache_tag}"
        for key, path in graph_files.items()
        if path
    }
    return result

@app.route("/api/video-analysis/upload", methods=["POST"])
def api_video_analysis_upload():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided."}), 400

    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "Please choose a video file."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"status": "error", "message": "Invalid file name."}), 400

    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        return jsonify({"status": "error", "message": "Unsupported video format."}), 400

    job_id = uuid.uuid4().hex[:12]
    stored_name = f"{job_id}_{safe_name}"
    input_path = os.path.join(VIDEO_ANALYSIS_UPLOAD_DIR, stored_name)
    file.save(input_path)

    job = {
        "id": job_id,
        "status": "queued",
        "original_name": file.filename,
        "input_name": stored_name,
        "input_path": input_path,
        "output_video": "",
        "summary": {},
        "error": "",
        "created_at": time.time(),
        "updated_at": time.time()
    }
    with data_lock:
        video_analysis_jobs[job_id] = job

    worker = threading.Thread(target=process_uploaded_video_job, args=(job_id,), daemon=True)
    worker.start()
    add_system_event(f"Queued video analysis: {file.filename}")
    return jsonify({
        "status": "success",
        "message": f"Uploaded {file.filename}. Analysis started.",
        "job": serialize_video_analysis_job(job)
    })

@app.route("/api/video-analysis/jobs")
def api_video_analysis_jobs():
    with data_lock:
        jobs = [serialize_video_analysis_job(job) for job in sorted(video_analysis_jobs.values(), key=lambda item: item.get("created_at", 0), reverse=True)]
    running_count = sum(1 for job in jobs if job.get("status") in {"queued", "running"})
    return jsonify({"jobs": jobs, "running_count": running_count})

@app.route("/video-analysis/results/<path:filename>")
def video_analysis_results_file(filename):
    return send_from_directory(VIDEO_ANALYSIS_RESULT_DIR, filename, as_attachment=False)

@app.route("/video-analysis/graphs/<path:filename>")
def video_analysis_graph_file(filename):
    return send_from_directory(VIDEO_ANALYSIS_GRAPH_DIR, filename, as_attachment=False)

def generate_video_file_mjpeg(video_path, playback_fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    source_fps = float(playback_fps or 0.0)
    if source_fps <= 0:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = STREAM_MAX_FPS
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = resize_frame_to_width(frame, STREAM_MAX_WIDTH)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(1.0 / max(1.0, source_fps))
    finally:
        cap.release()

@app.route("/video-analysis/stream/<job_id>")
def video_analysis_stream(job_id):
    with data_lock:
        job = dict(video_analysis_jobs.get(job_id) or {})
    video_path = job.get("output_video")
    if not video_path or not os.path.exists(video_path):
        return "Annotated video not found.", 404
    playback_fps = 0.0
    summary = job.get("summary") or {}
    try:
        playback_fps = float(summary.get("source_fps") or 0.0)
    except Exception:
        playback_fps = 0.0
    return Response(
        generate_video_file_mjpeg(video_path, playback_fps=playback_fps),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video-analysis/view/<job_id>")
def video_analysis_view(job_id):
    with data_lock:
        job = dict(video_analysis_jobs.get(job_id) or {})
    if not job:
        return "Video analysis job not found.", 404
    if not job.get("output_video") or not os.path.exists(job.get("output_video")):
        return "Annotated video not found.", 404

    title = job.get("original_name", "Annotated Video")
    stream_url = f"/video-analysis/stream/{job_id}"
    download_url = f"/video-analysis/results/{os.path.basename(job.get('output_video'))}"
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Annotated Video Viewer</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
            }}
            .shell {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 24px;
            }}
            .card {{
                background: #111827;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 18px 50px rgba(0,0,0,0.28);
            }}
            .video-frame {{
                width: 100%;
                border-radius: 14px;
                background: #020617;
                display: block;
            }}
            .actions {{
                display: flex;
                gap: 12px;
                margin-top: 16px;
                flex-wrap: wrap;
            }}
            .btn {{
                color: white;
                text-decoration: none;
                background: #2563eb;
                padding: 10px 16px;
                border-radius: 999px;
                font-weight: 700;
            }}
            .btn.secondary {{
                background: #334155;
            }}
            .meta {{
                color: #94a3b8;
                margin-top: 8px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="card">
                <h2 style="margin-top:0;">Annotated Video Viewer</h2>
                <div class="meta">{title}</div>
                <img class="video-frame" src="{stream_url}" alt="Annotated video stream">
                <div class="actions">
                    <a class="btn" href="/">Back to Dashboard</a>
                    <a class="btn secondary" href="{download_url}" target="_blank">Open Raw Saved File</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def run_server():
    # Runs Flask server in background thread
    try:
        print("Flask: Starting server...")
        app.run(
            host=settings["server_bind_host"],
            port=int(settings["server_port"]),
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Flask Error: {e}")
        import traceback
        traceback.print_exc()

print("Starting Flask server thread...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(1)  # Give Flask time to start
print(f"✓ Flask server should be running at http://{settings['server_bind_host']}:{settings['server_port']}")
print(f"  Access the report at: http://127.0.0.1:{settings['server_port']}/")
print(f"Deployment mode: {settings['deployment_mode']} | Node ID: {settings['node_id']}")
if is_edge_mode():
    print(f"Central server URL: {get_central_server_url()}")
if IDENTITY_MODE == "tracker":
    print("Identity mode: tracker-only numbering (ReID disabled).")
if not FACE_RECOGNITION_AVAILABLE:
    print("Face registration mode: disabled.")
print(f"Preview window: {PREVIEW_WINDOW_NAME}")

# ==================== YOLO11 Pose Model ====================
model = YOLO("yolo11n-pose.pt")  # nano pose model for primary live monitoring
try:
    model.to(YOLO_DEVICE)
except Exception:
    pass
fall_cooldown = {}  # Prevent spam alerts for same person

def set_video_analysis_job_state(job_id, **updates):
    with data_lock:
        job = video_analysis_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()

def append_video_analysis_event(events, timestamp_sec, event_name, detail=""):
    events.append({
        "time_sec": round(float(timestamp_sec), 2),
        "time_label": format_duration(float(timestamp_sec)),
        "event": event_name,
        "detail": detail
    })

def get_body_coverage_pct(confidences, threshold=0.5):
    if confidences is None or len(confidences) < 17:
        return 0.0
    key_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    detected = sum(1 for idx in key_indices if float(confidences[idx]) >= threshold)
    return (detected / float(len(key_indices))) * 100.0

def normalize_motion_for_fps(value, fps_hint=None):
    try:
        fps = float(fps_hint or MOTION_REFERENCE_FPS)
    except Exception:
        fps = MOTION_REFERENCE_FPS
    fps = max(1.0, fps)
    scale = min(4.0, max(0.5, fps / MOTION_REFERENCE_FPS))
    return float(value) * scale

def get_torso_angle(keypoints, conf):
    """Return torso angle plus shoulder/hip centers using COCO shoulders and hips."""
    required = [5, 6, 11, 12]
    if any(conf[idx] < 0.5 for idx in required):
        return None, None, None

    shoulder_center = np.array([
        (keypoints[5][0] + keypoints[6][0]) / 2.0,
        (keypoints[5][1] + keypoints[6][1]) / 2.0
    ], dtype=np.float32)
    hip_center = np.array([
        (keypoints[11][0] + keypoints[12][0]) / 2.0,
        (keypoints[11][1] + keypoints[12][1]) / 2.0
    ], dtype=np.float32)

    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return float(angle), shoulder_center, hip_center

def get_motion_score(current_center, previous_center, dt, fps_hint=None):
    """Smooth motion proxy in pixels per frame scaled for FPS-aware thresholds."""
    if current_center is None or previous_center is None or dt <= 0:
        return 0.0
    dist_per_second = np.linalg.norm(np.asarray(current_center) - np.asarray(previous_center)) / dt
    per_frame_motion = dist_per_second / max(1.0, float(fps_hint or MOTION_REFERENCE_FPS))
    return normalize_motion_for_fps(per_frame_motion, fps_hint=fps_hint)

def get_body_spread_ratio(keypoints, conf):
    """
    Compare the horizontal and vertical spread of major body keypoints.
    > 1 means the body is spread more horizontally than vertically.
    """
    indices = [5, 6, 11, 12, 13, 14, 15, 16]
    pts = [keypoints[idx] for idx in indices if conf[idx] >= 0.5]
    if len(pts) < 4:
        return 0.0
    pts = np.asarray(pts, dtype=np.float32)
    x_span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    y_span = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    return x_span / max(1.0, y_span)

def is_stable_horizontal_pose(
    torso_angle,
    aspect_ratio,
    body_spread_ratio,
    motion_score,
    angle_rate,
    stable_horizontal_time,
    recent_sudden_drop=False,
    transition_time=0.0
):
    """LYING requires a calm, horizontal, sustained posture."""
    if torso_angle is None and body_spread_ratio <= LYING_BODY_SPREAD_STABLE_RATIO and aspect_ratio < LYING_HORIZONTAL_BOX_RATIO:
        return False
    return (
        ((torso_angle is not None and torso_angle >= LYING_HORIZONTAL_ANGLE_MIN) or body_spread_ratio >= LYING_BODY_SPREAD_STABLE_RATIO)
        and (aspect_ratio >= LYING_HORIZONTAL_BOX_RATIO or body_spread_ratio >= LYING_BODY_SPREAD_STABLE_RATIO)
        and motion_score <= LYING_MAX_MOTION_SCORE
        and angle_rate <= LYING_MAX_ANGLE_RATE
        and stable_horizontal_time >= LYING_STABLE_SECONDS
        and transition_time >= LYING_TRANSITION_GRACE_SECONDS
        and not recent_sudden_drop
    )

def classify_pose(
    torso_angle,
    aspect_ratio,
    body_spread_ratio,
    motion_score,
    angle_rate,
    stable_horizontal_time,
    is_sitting=False,
    recent_sudden_drop=False,
    transition_time=0.0
):
    """Return high-level pose label before walking/fall refinements."""
    if is_stable_horizontal_pose(
        torso_angle,
        aspect_ratio,
        body_spread_ratio,
        motion_score,
        angle_rate,
        stable_horizontal_time,
        recent_sudden_drop=recent_sudden_drop,
        transition_time=transition_time
    ):
        return "LYING"

    if (
        (torso_angle is not None and torso_angle >= LYING_TRANSITION_ANGLE_MIN)
        or aspect_ratio >= LYING_TRANSITION_BOX_RATIO
        or body_spread_ratio >= LYING_BODY_SPREAD_TRANSITION_RATIO
    ):
        return "TRANSITION"

    if is_sitting:
        return "SITTING"

    return "STANDING"

def generate_video_analysis_graphs(job_id, original_name, summary):
    if not MATPLOTLIB_AVAILABLE:
        return {}

    safe_stem = secure_filename(os.path.splitext(original_name or job_id)[0]) or job_id
    activity_graph_path = os.path.join(VIDEO_ANALYSIS_GRAPH_DIR, f"{job_id}_{safe_stem}_activities.png")
    fall_graph_path = os.path.join(VIDEO_ANALYSIS_GRAPH_DIR, f"{job_id}_{safe_stem}_falls.png")

    durations_sec = [
        float(summary.get("walking_seconds", 0.0) or 0.0),
        float(summary.get("standing_seconds", 0.0) or 0.0),
        float(summary.get("sitting_seconds", 0.0) or 0.0),
        float(summary.get("sleeping_seconds", 0.0) or 0.0)
    ]
    activity_labels = ["Walking", "Standing", "Sitting", "Sleeping"]
    activity_colors = ["#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    bars = ax.bar(activity_labels, durations_sec, color=activity_colors)
    ax.set_title("Activity Duration Distribution")
    ax.set_ylabel("Duration (seconds)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, value in zip(bars, durations_sec):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(activity_graph_path, bbox_inches="tight")
    plt.close(fig)

    fall_labels = ["Minor Falls", "Major Falls", "Recoveries", "Multi-person Frames"]
    fall_values = [
        int(summary.get("minor_falls", 0) or 0),
        int(summary.get("major_falls", 0) or 0),
        int(summary.get("recoveries", 0) or 0),
        int(summary.get("multi_person_frames", 0) or 0)
    ]
    fall_colors = ["#f59e0b", "#ef4444", "#22c55e", "#06b6d4"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    bars = ax.bar(fall_labels, fall_values, color=fall_colors)
    ax.set_title("Fall And Scene Events")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, value in zip(bars, fall_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{value}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(fall_graph_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "activity_distribution": activity_graph_path,
        "fall_events": fall_graph_path
    }

def process_uploaded_video_job(job_id):
    with data_lock:
        job = dict(video_analysis_jobs.get(job_id) or {})
    if not job:
        return

    input_path = job.get("input_path")
    original_name = job.get("original_name", os.path.basename(input_path or "video"))
    set_video_analysis_job_state(job_id, status="running", error="")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        set_video_analysis_job_state(job_id, status="failed", error="Could not open uploaded video.")
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 20.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        ok, sample = cap.read()
        if not ok or sample is None:
            cap.release()
            set_video_analysis_job_state(job_id, status="failed", error="Uploaded video has no readable frames.")
            return
        frame_h, frame_w = sample.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_name = f"{job_id}_annotated.mp4"
    output_path = os.path.join(VIDEO_ANALYSIS_RESULT_DIR, output_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, source_fps, (frame_w, frame_h))

    activity_totals = {
        "WALKING": 0.0,
        "STANDING": 0.0,
        "SITTING": 0.0,
        "SLEEPING": 0.0
    }
    events = []
    current_state = "UNKNOWN"
    active_fall = None
    minor_fall_start_time = None
    lying_start_time = None
    recovery_confirm_count = 0
    last_pos = None
    velocity = 0.0
    vertical_velocity = 0.0
    previous_torso_angle = None
    previous_torso_time = None
    stable_horizontal_start_time = None
    transition_start_time = None
    recent_sudden_drop_until = 0.0
    prev_time_sec = 0.0
    processed_frames = 0
    multi_person_frames = 0
    valid_pose_frames = 0
    pose_confidence_sum = 0.0
    body_coverage_sum = 0.0
    processing_started = time.time()
    confirm_window = float(settings.get("fall_confirm_window_sec", 10.0))
    recovery_needed = max(10, int(source_fps * 0.8))

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_time_sec = processed_frames / source_fps if source_fps > 0 else processed_frames * 0.05
            results = model.track(
                frame,
                persist=True,
                conf=OFFLINE_DETECTION_CONFIDENCE,
                imgsz=OFFLINE_INFERENCE_IMGSZ,
                verbose=False
            )
            detection_count = int(len(results[0].boxes.id)) if (
                results and results[0].boxes is not None and getattr(results[0].boxes, "id", None) is not None
            ) else 0
            has_pose = bool(
                results
                and results[0].keypoints is not None
                and results[0].boxes is not None
                and getattr(results[0].boxes, "id", None) is not None
                and detection_count > 0
            )

            # Offline evaluation should be more forgiving than live view.
            # If the first pass misses side-angle or partially occluded poses,
            # run a second pass with a lower threshold and larger input size.
            if not has_pose:
                fallback_results = model.track(
                    frame,
                    persist=True,
                    conf=OFFLINE_FALLBACK_DETECTION_CONFIDENCE,
                    imgsz=OFFLINE_FALLBACK_INFERENCE_IMGSZ,
                    verbose=False
                )
                fallback_count = int(len(fallback_results[0].boxes.id)) if (
                    fallback_results and fallback_results[0].boxes is not None and getattr(fallback_results[0].boxes, "id", None) is not None
                ) else 0
                fallback_has_pose = bool(
                    fallback_results
                    and fallback_results[0].keypoints is not None
                    and fallback_results[0].boxes is not None
                    and getattr(fallback_results[0].boxes, "id", None) is not None
                    and fallback_count > 0
                )
                if fallback_has_pose:
                    results = fallback_results
                    detection_count = fallback_count

            display_frame = frame.copy()
            if detection_count > 1:
                multi_person_frames += 1

            if results and results[0].keypoints is not None and results[0].boxes is not None and getattr(results[0].boxes, "id", None) is not None and detection_count > 0:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes_xyxy]
                best_idx = int(np.argmax(areas))
                box = boxes_xyxy[best_idx]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame_w, box[2]), min(frame_h, box[3])
                keypoints = results[0].keypoints.xy[best_idx].cpu().numpy()
                confidences = results[0].keypoints.conf[best_idx].cpu().numpy()
                center_coords = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                draw_detailed_pose_overlay(display_frame, keypoints, confidences)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                pose_confidence = get_pose_confidence(confidences)
                body_coverage_pct = get_body_coverage_pct(confidences)
                valid_pose_frames += 1
                pose_confidence_sum += pose_confidence
                body_coverage_sum += body_coverage_pct

                if last_pos is not None:
                    dist = np.sqrt((center_coords[0] - last_pos[0]) ** 2 + (center_coords[1] - last_pos[1]) ** 2)
                    v_dist = center_coords[1] - last_pos[1]
                    velocity = velocity * 0.8 + dist * 0.2
                    vertical_velocity = vertical_velocity * 0.8 + v_dist * 0.2
                last_pos = center_coords

                torso_angle, shoulder_center, hip_center = get_torso_angle(keypoints, confidences)
                angle_rate = 0.0
                if torso_angle is not None and previous_torso_angle is not None and previous_torso_time is not None:
                    dt_angle = max(1e-3, frame_time_sec - previous_torso_time)
                    angle_rate = abs(torso_angle - previous_torso_angle) / dt_angle
                if torso_angle is not None:
                    previous_torso_angle = torso_angle
                    previous_torso_time = frame_time_sec

                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                aspect_ratio = bw / float(bh)
                body_spread_ratio = get_body_spread_ratio(keypoints, confidences)
                normalized_velocity = normalize_motion_for_fps(velocity, fps_hint=source_fps)
                normalized_v_velocity = normalize_motion_for_fps(vertical_velocity, fps_hint=source_fps)
                recent_sudden_drop = False
                if normalized_v_velocity > SUDDEN_DROP_VERTICAL_SPEED or normalized_velocity > SUDDEN_DROP_MOTION_SCORE:
                    recent_sudden_drop_until = frame_time_sec + SUDDEN_DROP_HOLD_SECONDS
                if frame_time_sec < recent_sudden_drop_until:
                    recent_sudden_drop = True

                horizontal_candidate = (
                    torso_angle is not None
                    and (torso_angle >= LYING_TRANSITION_ANGLE_MIN or aspect_ratio >= LYING_TRANSITION_BOX_RATIO)
                )
                if horizontal_candidate:
                    if transition_start_time is None:
                        transition_start_time = frame_time_sec
                else:
                    transition_start_time = None

                stable_horizontal_candidate = (
                    torso_angle is not None
                    and torso_angle >= LYING_HORIZONTAL_ANGLE_MIN
                    and aspect_ratio >= LYING_HORIZONTAL_BOX_RATIO
                    and normalized_velocity <= LYING_MAX_MOTION_SCORE
                    and angle_rate <= LYING_MAX_ANGLE_RATE
                    and not recent_sudden_drop
                )
                if stable_horizontal_candidate:
                    if stable_horizontal_start_time is None:
                        stable_horizontal_start_time = frame_time_sec
                else:
                    stable_horizontal_start_time = None

                stable_horizontal_time = 0.0 if stable_horizontal_start_time is None else max(0.0, frame_time_sec - stable_horizontal_start_time)
                transition_time = 0.0 if transition_start_time is None else max(0.0, frame_time_sec - transition_start_time)
                activity = classify_activity(
                    keypoints,
                    confidences,
                    velocity=velocity,
                    v_velocity=vertical_velocity,
                    aspect_ratio=aspect_ratio,
                    fps_hint=source_fps,
                    angle_rate=angle_rate,
                    stable_horizontal_time=stable_horizontal_time,
                    transition_time=transition_time,
                    recent_sudden_drop=recent_sudden_drop,
                    body_spread_ratio=body_spread_ratio
                )
                new_state = current_state if activity == "UNKNOWN" else activity
                prev_state = current_state

                if prev_state == "MAJOR FALL" and new_state in ["MINOR FALL", "LYING"]:
                    new_state = "MAJOR FALL"
                if normalized_v_velocity < -1.0 and prev_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"] and new_state in ["MINOR FALL", "LYING"]:
                    new_state = "STANDING"
                if prev_state in ["LYING", "SLEEPING", "TRANSITION"] and new_state == "MINOR FALL":
                    new_state = "TRANSITION"
                if prev_state in ["TRANSITION", "LYING", "SLEEPING"] and new_state in ["STANDING", "WALKING"]:
                    if normalized_v_velocity > -1.2 and normalized_velocity <= LYING_MAX_MOTION_SCORE:
                        new_state = prev_state
                if prev_state == "TRANSITION" and transition_time >= DOWN_STATE_HOLD_SECONDS and normalized_velocity <= LYING_MAX_MOTION_SCORE:
                    new_state = "LYING"
                if prev_state in ["TRANSITION", "LYING", "SLEEPING"] and normalized_velocity <= LYING_MAX_MOTION_SCORE:
                    if transition_time >= SLEEPING_AFTER_LYING_SECONDS:
                        new_state = "SLEEPING"
                    elif transition_time >= DOWN_STATE_HOLD_SECONDS and new_state in ["TRANSITION", "STANDING", "WALKING"]:
                        new_state = "LYING"

                is_currently_down = new_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]
                if is_currently_down and active_fall is None and prev_state in ["WALKING", "STANDING", "RECOVERED"]:
                    if new_state in ["LYING", "SLEEPING"]:
                        new_state = "MINOR FALL"
                    active_fall = "MINOR"
                    append_video_analysis_event(events, frame_time_sec, "MINOR FALL", "Detected abnormal down posture.")

                if is_currently_down:
                    recovery_confirm_count = 0
                    if minor_fall_start_time is None and prev_state in ["WALKING", "STANDING", "RECOVERED"]:
                        minor_fall_start_time = frame_time_sec
                    if active_fall == "MINOR" and new_state in ["LYING", "MAJOR FALL"] and minor_fall_start_time is not None and (frame_time_sec - minor_fall_start_time) > confirm_window:
                        active_fall = "MAJOR"
                        append_video_analysis_event(events, frame_time_sec, "MAJOR FALL", f"No recovery for {confirm_window:.1f}s.")
                elif new_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"]:
                    recovery_confirm_count += 1
                    if recovery_confirm_count >= recovery_needed:
                        if active_fall is not None:
                            append_video_analysis_event(events, frame_time_sec, "RECOVERED", "Returned to stable upright posture.")
                            active_fall = None
                        minor_fall_start_time = None
                        lying_start_time = None
                        recovery_confirm_count = 0

                if new_state == "LYING":
                    if lying_start_time is None:
                        lying_start_time = frame_time_sec
                    if prev_state in ["SITTING", "SLEEPING", "UNKNOWN"] and normalized_v_velocity < 4.5:
                        new_state = "SLEEPING"
                    elif lying_start_time is not None and (frame_time_sec - lying_start_time) > SLEEPING_AFTER_LYING_SECONDS:
                        new_state = "SLEEPING"
                elif new_state != "SLEEPING":
                    lying_start_time = None

                duration = max(0.0, frame_time_sec - prev_time_sec)
                if prev_state in activity_totals:
                    activity_totals[prev_state] += duration

                current_state = new_state if new_state != "UNKNOWN" else current_state
                prev_time_sec = frame_time_sec

                color = (0, 0, 255) if "FALL" in current_state else ((0, 255, 0) if current_state == "RECOVERED" else (255, 255, 255))
                cv2.putText(display_frame, f"State: {current_state} | Conf {pose_confidence:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.putText(display_frame, f"W:{format_duration(activity_totals['WALKING'])} St:{format_duration(activity_totals['STANDING'])} S:{format_duration(activity_totals['SITTING'])} Sl:{format_duration(activity_totals['SLEEPING'])}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Body Coverage: {body_coverage_pct:.1f}%", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
                if detection_count > 1:
                    cv2.putText(display_frame, f"Multiple people visible: {detection_count}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                cv2.putText(display_frame, "No pose detected", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            writer.write(display_frame)
            processed_frames += 1
            if processed_frames % 25 == 0:
                set_video_analysis_job_state(
                    job_id,
                    status="running",
                    summary={
                        "frames_processed": processed_frames,
                        "processing_fps": round(processed_frames / max(0.001, time.time() - processing_started), 2),
                        "avg_frame_time_ms": round((1000.0 * max(0.001, time.time() - processing_started)) / max(1, processed_frames), 2),
                        "avg_pose_confidence": round(pose_confidence_sum / max(1, valid_pose_frames), 4),
                        "pose_coverage_pct": round((valid_pose_frames / max(1, processed_frames)) * 100.0, 2),
                        "body_coverage_pct": round(body_coverage_sum / max(1, valid_pose_frames), 2),
                        "walking_dur": format_duration(activity_totals["WALKING"]),
                        "standing_dur": format_duration(activity_totals["STANDING"]),
                        "sitting_dur": format_duration(activity_totals["SITTING"]),
                        "sleeping_dur": format_duration(activity_totals["SLEEPING"]),
                        "minor_falls": sum(1 for event in events if event["event"] == "MINOR FALL"),
                        "major_falls": sum(1 for event in events if event["event"] == "MAJOR FALL"),
                        "events": events[-5:],
                        "multi_person_frames": multi_person_frames,
                        "multi_person_detection_coverage_pct": round((multi_person_frames / max(1, processed_frames)) * 100.0, 2),
                        "fall_detection_rate_pct": None,
                        "false_alerts": None
                    }
                )
    except Exception as e:
        cap.release()
        writer.release()
        set_video_analysis_job_state(job_id, status="failed", error=f"Analysis failed: {e}")
        add_system_event(f"Video analysis failed for {original_name}: {e}", level="error")
        return

    cap.release()
    writer.release()

    total_runtime = max(0.001, time.time() - processing_started)
    total_video_sec = processed_frames / source_fps if source_fps > 0 else 0.0
    if current_state in activity_totals and total_video_sec > prev_time_sec:
        activity_totals[current_state] += max(0.0, total_video_sec - prev_time_sec)

    summary = {
        "frames_processed": processed_frames,
        "video_duration_sec": round(total_video_sec, 2),
        "video_duration_dur": format_duration(total_video_sec),
        "processing_fps": round(processed_frames / total_runtime, 2),
        "avg_frame_time_ms": round((1000.0 * total_runtime) / max(1, processed_frames), 2),
        "source_fps": round(source_fps, 2),
        "valid_pose_frames": valid_pose_frames,
        "avg_pose_confidence": round(pose_confidence_sum / max(1, valid_pose_frames), 4),
        "pose_coverage_pct": round((valid_pose_frames / max(1, processed_frames)) * 100.0, 2),
        "body_coverage_pct": round(body_coverage_sum / max(1, valid_pose_frames), 2),
        "walking_seconds": round(activity_totals["WALKING"], 2),
        "standing_seconds": round(activity_totals["STANDING"], 2),
        "sitting_seconds": round(activity_totals["SITTING"], 2),
        "sleeping_seconds": round(activity_totals["SLEEPING"], 2),
        "walking_dur": format_duration(activity_totals["WALKING"]),
        "standing_dur": format_duration(activity_totals["STANDING"]),
        "sitting_dur": format_duration(activity_totals["SITTING"]),
        "sleeping_dur": format_duration(activity_totals["SLEEPING"]),
        "minor_falls": sum(1 for event in events if event["event"] == "MINOR FALL"),
        "major_falls": sum(1 for event in events if event["event"] == "MAJOR FALL"),
        "recoveries": sum(1 for event in events if event["event"] == "RECOVERED"),
        "multi_person_frames": multi_person_frames,
        "multi_person_detection_coverage_pct": round((multi_person_frames / max(1, processed_frames)) * 100.0, 2),
        "fall_detection_rate_pct": None,
        "false_alerts": None,
        "events": events
    }
    graph_files = generate_video_analysis_graphs(job_id, original_name, summary)
    set_video_analysis_job_state(job_id, status="completed", output_video=output_path, summary=summary, graph_files=graph_files)
    add_system_event(f"Video analysis completed: {original_name}")

def classify_activity(
    keypoints,
    conf,
    velocity=0,
    v_velocity=0,
    aspect_ratio=1.0,
    fps_hint=MOTION_REFERENCE_FPS,
    angle_rate=0.0,
    stable_horizontal_time=0.0,
    transition_time=0.0,
    recent_sudden_drop=False,
    body_spread_ratio=0.0
):
    """
    Advanced skeleton-based activity classification with Impact Detection.
    """
    try:
        # Check confidence of critical joints
        critical_joints = [5, 6, 11, 12] # Shoulders and Hips

        angle, shoulder_center, hip_center = get_torso_angle(keypoints, conf)
        if angle is None:
            if conf[0] > 0.5 and (conf[11] > 0.5 or conf[12] > 0.5):
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (conf[11] > 0.5 and conf[12] > 0.5) else (keypoints[11][1] if conf[11] > 0.5 else keypoints[12][1])
                if keypoints[0][1] > hip_y - 10:
                    return "LYING"
            if aspect_ratio > 1.8:
                return "LYING"
            return "UNKNOWN"

        sho_y = shoulder_center[1]
        sho_x = shoulder_center[0]
        hip_y = hip_center[1]
        hip_x = hip_center[0]
        dy = hip_y - sho_y
        dx = hip_x - sho_x
        adj_velocity = normalize_motion_for_fps(velocity, fps_hint=fps_hint)
        adj_v_velocity = normalize_motion_for_fps(v_velocity, fps_hint=fps_hint)

        # Impact Detection: High vertical velocity (downward) + High angle or wide box
        # Threshold: > 8 pixels/frame downward is usually a fall
        if adj_v_velocity > 8.0 and (angle > 30 or aspect_ratio > 1.2):
            return "MAJOR FALL"

        # Transitional fall cue: only treat the posture as a fall when there is
        # meaningful downward motion. This avoids slow intentional sit/lie
        # transitions being labeled as MINOR FALL.
        if adj_v_velocity > 4.5 and (angle > 35 or aspect_ratio > 1.3):
            return "MINOR FALL"

        # 3. SITTING vs UPRIGHT (STANDING/WALKING)
        # Move sitting logic up to prevent "sitting on floor" from being "Minor Fall"
        is_sitting = False
        if conf[13] > 0.5 and conf[14] > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            if conf[15] > 0.5 and conf[16] > 0.5:
                ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
                upper_leg = abs(knee_y - hip_y)
                lower_leg = abs(ank_y - knee_y)
                if upper_leg < lower_leg * 0.5: 
                    is_sitting = True
        
        if not is_sitting and conf[15] > 0.5 and conf[16] > 0.5:
            torso_len = np.sqrt(dx**2 + dy**2)
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            total_h = abs(ank_y - sho_y)
            # Sitting on floor with spread legs often results in torso_len / total_h > 0.6
            if torso_len / total_h > 0.6: 
                is_sitting = True
        
        # Additional floor sitting check: Hips close to floor (ankles) but torso not horizontal
        if not is_sitting and conf[11] > 0.5 and conf[12] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            hip_to_floor = abs(ank_y - hip_y)
            torso_len = np.sqrt(dx**2 + dy**2)
            if hip_to_floor < torso_len * 0.4 and angle < 50:
                is_sitting = True

        # Nose-to-ground ratio: If head is significantly lower than normal relative to torso
        # and the body has settled, it supports LYING rather than SITTING.
        if conf[0] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            nose_y = keypoints[0][1]
            head_height = abs(ank_y - nose_y)
            torso_len = np.sqrt(dx**2 + dy**2)
            if head_height < torso_len * 0.8:
                stable_horizontal_time = max(stable_horizontal_time, LYING_STABLE_SECONDS)

        pose_phase = classify_pose(
            angle,
            aspect_ratio,
            body_spread_ratio,
            adj_velocity,
            angle_rate,
            stable_horizontal_time,
            is_sitting=is_sitting,
            recent_sudden_drop=recent_sudden_drop,
            transition_time=transition_time
        )
        if (
            pose_phase == "TRANSITION"
            and body_spread_ratio >= LYING_BODY_SPREAD_STABLE_RATIO
            and aspect_ratio >= 0.95
            and adj_velocity <= LYING_MAX_MOTION_SCORE
            and transition_time >= LYING_STABLE_SECONDS
            and not recent_sudden_drop
        ):
            pose_phase = "LYING"
        elif (
            pose_phase == "TRANSITION"
            and transition_time >= LYING_FROM_TRANSITION_SECONDS
            and adj_velocity <= LYING_MAX_MOTION_SCORE
            and angle_rate <= LYING_MAX_ANGLE_RATE
            and not recent_sudden_drop
            and not is_sitting
        ):
            pose_phase = "LYING"
        if pose_phase in ["LYING", "TRANSITION", "SITTING"]:
            return pose_phase
        
        # 2. MINOR FALL (Significant tilt but not fully flat)
        # Refined: Increased angle/aspect ratio and ensure not actively walking
        if (angle > 45 or aspect_ratio > 1.5) and adj_velocity < 3.0 and adj_v_velocity > 2.5:
            return "MINOR FALL"
        
        # 4. STANDING vs WALKING (Using velocity + Pose)
        # Increased threshold to 3.0 for WALKING to reduce noise
        if adj_velocity > 5.0 and body_spread_ratio < 1.0 and aspect_ratio < 1.05:
            return "WALKING"
        
        # Fallback to pose-based walking detection if velocity is moderate
        if adj_velocity > 1.5 and conf[15] > 0.5 and conf[16] > 0.5 and body_spread_ratio < 1.0 and not recent_sudden_drop:
            feet_dist = abs(keypoints[15][0] - keypoints[16][0])
            shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
            # Feet must be wider than shoulders to be 'walking' if velocity is low
            if feet_dist > shoulder_width * 1.2 and angle < 35:
                return "WALKING"

        return "STANDING"

    except Exception:
        return "UNKNOWN"

def send_fall_alert(alert_msg, pid, fall_type, coords=None):
    global last_global_alert_time, last_alert_coords, last_alert_pid, status_message, status_expiry
    try:
        now = time.time()
        
        # Spatial-Temporal Squelch:
        # If we sent an alert of this type recently (< 5s) and it was in the same area (< 150px)
        # then it's likely a phantom ID/tracker drift for the same person.
        if coords and fall_type in last_alert_coords:
            prev_coords = last_alert_coords[fall_type]
            dist = np.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2)
            if dist < 150 and (now - last_global_alert_time) < 5:
                # If it's the SAME person (resolved name), definitely skip
                # If it's a DIFFERENT person but very close/recent, it's likely a ghost ID
                print(f"🤫 Squelching redundant {fall_type} for {pid} (likely ghost ID)")
                return
        
        # Update last alert state
        last_global_alert_time = now
        if coords: last_alert_coords[fall_type] = coords
        last_alert_pid[fall_type] = pid

        # Update status message for visual feedback on frame
        status_message = alert_msg
        status_expiry = now + 5

        print(f"⚠️  {alert_msg}! Sending alert...")
        increment_evaluation_metric("fall_alerts_total")
        if fall_type == "MINOR FALL":
            increment_evaluation_metric("minor_falls_total")
        elif fall_type == "MAJOR FALL":
            increment_evaluation_metric("major_falls_total")
        elif fall_type == "RECOVERED":
            increment_evaluation_metric("recoveries_total")
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO falls (timestamp, person_id, type, unix_timestamp) VALUES (?, ?, ?, ?)",
                  (datetime.now(), str(pid), fall_type, now))
        conn.commit()
        conn.close()

        if is_edge_mode():
            requests.post(
                f"{get_central_server_url()}/trigger",
                json={"person_id": str(pid), "message": alert_msg, "type": fall_type},
                timeout=2
            )
        else:
            requests.post(
                f"http://127.0.0.1:{settings['server_port']}/trigger",
                json={"person_id": str(pid), "message": alert_msg, "type": fall_type},
                timeout=1
            )

        telegram_text = f"{pid}: {fall_type}\n{alert_msg}"
        telegram_category = f"fall:{fall_type.lower()}"
        if fall_type == "MAJOR FALL":
            send_telegram_burst_async(
                telegram_text,
                category=telegram_category,
                count=MAJOR_FALL_TELEGRAM_BURST_COUNT,
                delay_sec=MAJOR_FALL_TELEGRAM_BURST_DELAY_SEC,
                force=True
            )
            add_system_event(
                f"Major fall burst queued: {MAJOR_FALL_TELEGRAM_BURST_COUNT} Telegram notifications"
            )
        else:
            send_telegram_message(telegram_text, category=telegram_category)
        print("✓ Fall alert sent and logged successfully")
    except Exception as e:
        print(f"✗ Failed to send/log fall alert: {e}")

def post_to_central(path, payload, timeout=2):
    if not is_edge_mode():
        return False
    try:
        response = requests.post(f"{get_central_server_url()}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        increment_evaluation_metric("central_sync_ok")
        return True
    except Exception as e:
        increment_evaluation_metric("central_sync_fail")
        add_system_event(f"Central sync failed for {path}: {e}", level="warning")
        return False

def send_node_heartbeat():
    payload = {
        "node_id": settings["node_id"],
        "deployment_mode": settings["deployment_mode"],
        "camera_available": camera_available,
        "has_frame": latest_stream_frame is not None
    }
    return post_to_central("/api/node-heartbeat", payload, timeout=2)

def send_edge_report_snapshot():
    snapshot = build_report_snapshot(include_remote=False)
    snapshot.update({
        "node_id": settings["node_id"],
        "camera_available": camera_available,
        "has_frame": latest_stream_frame is not None
    })
    return post_to_central("/api/edge/report", snapshot, timeout=3)

# ==================== Camera Loop ====================
def open_camera():
    """Robust camera opener for Windows"""
    try:
        camera_index = int(settings.get("preferred_camera", "0"))
    except Exception:
        camera_index = 0
    for _ in range(3): # Try up to 3 times
        # Using DSHOW (DirectShow) on Windows is much more stable for resolution changes
        c = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera_index)
        if c.isOpened():
            # Set buffer size to 1 for lowest latency
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
            return c
        time.sleep(0.5)
    return None

def resize_frame_to_width(frame, max_width):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    target_size = (max_width, max(1, int(h * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def update_stream_frame(frame):
    global latest_stream_frame, last_stream_update_at
    if frame is None:
        return
    try:
        now = time.perf_counter()
        if (now - last_stream_update_at) < (1.0 / STREAM_MAX_FPS):
            return
        stream_frame = resize_frame_to_width(frame, STREAM_MAX_WIDTH)
        ok, encoded = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
        if ok:
            with data_lock:
                latest_stream_frame = encoded.tobytes()
            last_stream_update_at = now
    except Exception:
        pass

def clear_stream_frame():
    global latest_stream_frame
    with data_lock:
        latest_stream_frame = None

def show_preview_window(frame):
    global preview_window_enabled, preview_window_initialized, last_preview_update_at
    if not preview_window_enabled:
        return
    if frame is None:
        return
    try:
        if not preview_window_initialized:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, PREVIEW_WINDOW_WIDTH, PREVIEW_WINDOW_HEIGHT)
            preview_window_initialized = True
        now = time.perf_counter()
        if (now - last_preview_update_at) >= (1.0 / PREVIEW_MAX_FPS):
            cv2.imshow(PREVIEW_WINDOW_NAME, frame)
            last_preview_update_at = now
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            preview_window_enabled = False
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)
            preview_window_initialized = False
    except Exception:
        pass

def draw_detailed_pose_overlay(frame, keypoints, confidences):
    if frame is None or keypoints is None or confidences is None:
        return

    # More complete COCO-style body layout with face, arms, torso, and legs.
    face_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]
    upper_body_connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
    torso_connections = [(5, 11), (6, 12), (11, 12)]
    lower_body_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]
    all_connections = (
        face_connections
        + upper_body_connections
        + torso_connections
        + lower_body_connections
    )

    for start_idx, end_idx in all_connections:
        if confidences[start_idx] > 0.35 and confidences[end_idx] > 0.35:
            pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            if start_idx <= 4 and end_idx <= 4:
                color = (255, 180, 0)
                thickness = 1
            elif start_idx in [5, 6, 7, 8, 9, 10] or end_idx in [5, 6, 7, 8, 9, 10]:
                color = (80, 220, 120)
                thickness = 2
            elif start_idx in [11, 12, 13, 14, 15, 16] or end_idx in [11, 12, 13, 14, 15, 16]:
                color = (80, 160, 255)
                thickness = 2
            else:
                color = (200, 200, 200)
                thickness = 2

            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw joint markers with varied sizes for a more readable skeleton.
    for joint_idx in range(len(keypoints)):
        conf = confidences[joint_idx]
        if conf <= 0.35:
            continue

        point = (int(keypoints[joint_idx][0]), int(keypoints[joint_idx][1]))
        if joint_idx == 0:
            color = (0, 255, 255)
            radius = 5
        elif joint_idx in [5, 6, 11, 12]:
            color = (0, 255, 0)
            radius = 5
        elif joint_idx in [7, 8, 13, 14]:
            color = (255, 200, 0)
            radius = 4
        elif joint_idx in [9, 10, 15, 16]:
            color = (255, 120, 120)
            radius = 4
        else:
            color = (255, 255, 255)
            radius = 3

        cv2.circle(frame, point, radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    # YOLO pose gives wrist points, not full hand landmarks. Add a small inferred
    # hand marker beyond each wrist so hands are easier to see in the overlay.
    hand_pairs = [(7, 9), (8, 10)]  # elbow -> wrist
    for elbow_idx, wrist_idx in hand_pairs:
        if confidences[elbow_idx] <= 0.35 or confidences[wrist_idx] <= 0.35:
            continue

        elbow = keypoints[elbow_idx]
        wrist = keypoints[wrist_idx]
        hand_vec = wrist - elbow
        hand_len = np.linalg.norm(hand_vec)
        if hand_len <= 1:
            continue

        unit_vec = hand_vec / hand_len
        hand_point = wrist + unit_vec * min(18.0, hand_len * 0.35)
        wrist_pt = (int(wrist[0]), int(wrist[1]))
        hand_pt = (int(hand_point[0]), int(hand_point[1]))

        cv2.line(frame, wrist_pt, hand_pt, (255, 170, 80), 2, cv2.LINE_AA)
        cv2.circle(frame, hand_pt, 6, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, hand_pt, 4, (255, 210, 120), -1, cv2.LINE_AA)

def enter_low_power_mode(frame, frame_width):
    global system_sleeping, preview_window_initialized
    sleep_notice_frame = frame.copy()
    cv2.rectangle(sleep_notice_frame, (0, 0), (frame_width, 84), (0, 140, 255), -1)
    cv2.putText(
        sleep_notice_frame,
        "WARD OUT OF FRAME.",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )
    cv2.putText(
        sleep_notice_frame,
        "GOING INTO LOW POWER MODE.",
        (10, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 0, 0),
        2
    )
    update_stream_frame(sleep_notice_frame)
    show_preview_window(sleep_notice_frame)
    add_system_event("WARD OUT OF FRAME. GOING INTO LOW POWER MODE.", level="warning")
    print(f"💤 Low Power Mode: No person detected for {LOW_POWER_IDLE_TIMEOUT_SEC:.1f}s. Turning off webcam...")
    increment_evaluation_metric("low_power_entries")
    update_evaluation_metric("last_sleep_started_at", time.time())
    system_sleeping = True
    cv2.destroyAllWindows()
    preview_window_initialized = False

cap = open_camera()
camera_available = cap is not None
if not camera_available:
    print("Warning: Cannot open camera. Starting in dashboard-only mode.")
    add_system_event("Camera unavailable. Running in dashboard-only mode.", level="warning")

frame_count = 0
start_time = time.time()
last_detection = {}  # Track last frame when person was detected
last_motion_time = time.time()
out_of_frame_since = None
prev_gray = None
system_sleeping = False
wake_grace_until = 0.0
last_node_heartbeat_at = 0
last_edge_sync_at = 0

while not shutdown_event.is_set():
    frame_count += 1
    now = time.time()
    frame = None
    just_woke_up = False

    if is_edge_mode() and (now - last_node_heartbeat_at) >= 5:
        send_node_heartbeat()
        last_node_heartbeat_at = now

    if not camera_available:
        time.sleep(1.0)
        continue
    
    # --- 1. Handle Sleep Mode Lifecycle ---
    if system_sleeping:
        time.sleep(LOW_POWER_PEEK_INTERVAL_SEC)
        cap = open_camera()
        if cap is None: continue
        
        # Peek at low resolution to save power
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, peek_frame = cap.read()
        
        if not ret or peek_frame is None:
            if cap: cap.release()
            cap = None
            continue
            
        gray = cv2.cvtColor(peek_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            if np.sum(cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]) > 15000:
                print("☀️ Motion detected! Webcam and AI Reopened.")
                system_sleeping = False
                camera_available = True
                last_motion_time = now
                out_of_frame_since = None
                wake_grace_until = now + WAKE_GRACE_PERIOD_SEC
                with data_lock:
                    sleep_started_at = evaluation_metrics.get("last_sleep_started_at")
                if sleep_started_at:
                    record_wake_latency(max(0.0, now - sleep_started_at))
                    update_evaluation_metric("last_sleep_started_at", None)
                # Restore full resolution for AI
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
                # Grab a fresh full-res frame and resume immediately.
                ret, frame = cap.read()
                if not ret: continue
                just_woke_up = True
                prev_gray = None
                update_stream_frame(frame)
                show_preview_window(frame)
            else:
                prev_gray = gray
                if cap: cap.release()
                cap = None
                continue
        else:
            prev_gray = gray
            if cap: cap.release()
            cap = None
            continue

    # --- 2. Normal Camera Operation ---
    if not system_sleeping:
        if cap is None or not cap.isOpened():
            cap = open_camera()
            if cap is None:
                camera_available = False
                time.sleep(1)
                continue
            camera_available = True

        if frame is None:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Camera read failed. Retrying...")
                if cap: cap.release()
                cap = None
                with data_lock:
                    latest_stream_frame = None
                continue
        
        # Motion detection to keep system awake
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if just_woke_up:
            last_motion_time = now
            prev_gray = gray
        elif prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            if np.sum(cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]) > 20000:
                last_motion_time = now
            prev_gray = cv2.addWeighted(prev_gray, 0.9, gray, 0.1, 0)
        else:
            prev_gray = gray

    last_frame = frame.copy() # Store for registration
    try:
        h, w = frame.shape[:2]
        # Make a copy to display
        display_frame = frame.copy()
        latest_person_crops.clear()
        latest_person_pose_meta.clear()

        # Run YOLO tracking with higher confidence to reduce false positives
        if settings.get("enable_detection", True):
            results = model.track(
                frame,
                persist=True,
                conf=LIVE_DETECTION_CONFIDENCE,
                imgsz=LIVE_INFERENCE_IMGSZ,
                verbose=False
            )
        else:
            results = [type("EmptyResult", (), {"keypoints": None, "boxes": type("EmptyBoxes", (), {"id": None})()})()]

        detected_ids = set()  # Track which people are in this frame
        person_visible_this_frame = bool(
            results[0].boxes is not None
            and getattr(results[0].boxes, "id", None) is not None
            and len(results[0].boxes.id) > 0
        )
        visible_people_count = int(len(results[0].boxes.id)) if (
            results[0].boxes is not None
            and getattr(results[0].boxes, "id", None) is not None
        ) else 0
        multi_person_visible = bool(
            results[0].boxes is not None
            and getattr(results[0].boxes, "id", None) is not None
            and len(results[0].boxes.id) > 1
        )
        should_send_multi_person_notice = False
        should_refresh_single_person_dashboard = False
        with data_lock:
            prev_multi_person_scene = bool(multi_person_scene_active)
            multi_person_scene_active = bool(multi_person_visible)
            multi_person_scene_count = int(visible_people_count)
            if multi_person_visible:
                last_notified_activity.clear()
                if not prev_multi_person_scene:
                    should_send_multi_person_notice = True
            else:
                if prev_multi_person_scene:
                    should_refresh_single_person_dashboard = True
                multi_person_notice_sent = False
        update_evaluation_metric("visible_people_count", int(visible_people_count))
        update_evaluation_metric("multi_person_mode", 1 if multi_person_visible else 0)
        if should_send_multi_person_notice:
            notify_multi_person_scene()
        elif should_refresh_single_person_dashboard and telegram_ready():
            upsert_telegram_dashboard(force_send=True)
        
        if results[0].keypoints is not None and results[0].boxes.id is not None:
            # Keep system awake if people are detected
            last_motion_time = now
            max_people = int(settings.get("max_people_to_track", DEFAULT_SETTINGS["max_people_to_track"]))
            if multi_person_visible:
                max_people = max(max_people, MULTI_PERSON_MIN_TRACK_COUNT)
            for i, kp in enumerate(results[0].keypoints.xy[:max_people]):
                # 1. Get YOLO track ID
                yolo_id = int(results[0].boxes.id[i])
                yolo_id_str = str(yolo_id)
                
                # Bounding box for movement check
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                person_img = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                center_coords = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if multi_person_visible:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                # 2. Map YOLO ID to Persistent ID (ReID)
                if yolo_id_str not in tracker_to_persistent:
                    if IDENTITY_MODE == "tracker":
                        persistent_id = yolo_id_str
                        tracker_to_persistent[yolo_id_str] = persistent_id
                        person_start_pos[persistent_id] = center_coords
                        person_frames_seen[persistent_id] = 0
                        person_is_confirmed[persistent_id] = False
                    else:
                        if x2 > x1 and y2 > y1:
                            embedding = reid_manager.get_embedding(person_img)
                            # Stability Patch: Fallback to Clothing Color
                            current_sig = get_color_signature(person_img)
                            
                            with data_lock:
                                persistent_id = reid_manager.match_identity(embedding, current_sig=current_sig)
                            tracker_to_persistent[yolo_id_str] = persistent_id
                            
                            # Initialize movement tracking
                            person_start_pos[persistent_id] = center_coords
                            person_frames_seen[persistent_id] = 0
                            person_is_confirmed[persistent_id] = False
                        else:
                            continue
                
                persistent_id = tracker_to_persistent[yolo_id_str]
                if person_img is not None and person_img.size > 0:
                    latest_person_crops[str(persistent_id)] = person_img.copy()
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                keypoints = kp.cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                full_body_visible, visibility_message = evaluate_full_body_visibility(confidences)
                registration_visible, registration_message = evaluate_registration_capture_visibility(confidences)
                latest_person_pose_meta[str(persistent_id)] = {
                    "full_body_visible": full_body_visible,
                    "message": visibility_message,
                    "registration_visible": registration_visible,
                    "registration_message": registration_message
                }

                ward_profile_exists = ward_profile_ready()
                registration_target = str(ward_registration_session.get("target_id") or "")

                is_ward_detection = not ward_profile_exists
                if ward_profile_exists:
                    is_ward_detection = str(persistent_id) == str(ward_locked_persistent_id)
                    if not is_ward_detection and person_img is not None:
                        matched_ward, ward_score = match_ward_profile(person_img)
                        if matched_ward:
                            if str(ward_locked_persistent_id) != str(persistent_id):
                                add_system_event(
                                    f"Ward {ward_profile.get('name', SINGLE_PERSON_LABEL)} recognized again (score {ward_score:.2f})"
                                )
                            ward_locked_persistent_id = str(persistent_id)
                            monitored_persistent_id = str(persistent_id)
                            manual_id_map[str(persistent_id)] = ward_profile.get("name", SINGLE_PERSON_LABEL)
                            save_manual_id_map()
                            is_ward_detection = True
                            add_ward_signature_sample(
                                embedding=reid_manager.get_embedding(person_img),
                                color_sig=get_color_signature(person_img)
                            )

                    should_enforce_ward_only = ward_registration_session.get("active") or pending_registration.get("active")
                    if should_enforce_ward_only and not is_ward_detection and registration_target != str(persistent_id):
                        cv2.putText(
                            display_frame,
                            "Guest / Non-ward",
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            2
                        )
                        continue
                
                # IMPORTANT: Always mark as detected so they aren't pruned while being confirmed
                person_frames_seen[persistent_id] = person_frames_seen.get(persistent_id, 0) + 1

                # 3. Static Object Filtering (e.g., clothes on wall)
                # If a person hasn't moved at all in 60 frames (~2s), it's likely a static object
                is_static_object = False
                if not person_is_confirmed.get(persistent_id, False):
                    start_pos = person_start_pos.get(persistent_id, center_coords)
                    dist = np.sqrt((center_coords[0]-start_pos[0])**2 + (center_coords[1]-start_pos[1])**2)
                    
                    if dist > 30: # Moved 30 pixels? Confirmed human
                        person_is_confirmed[persistent_id] = True
                    elif person_frames_seen[persistent_id] > 60: 
                        is_static_object = True
                
                # Draw skeleton for ALL detections (including unconfirmed) so user sees tracking
                # --- Draw Detailed Pose Overlay ---
                draw_detailed_pose_overlay(display_frame, keypoints, confidences)
                pose_confidence = get_pose_confidence(confidences)
                if auto_capture_ward_registration_sample(persistent_id, person_img):
                    status_message = f"Building gallery for {ward_registration_session.get('name', 'ward')} | {ward_registration_session.get('captures', 0)} samples saved"
                    status_expiry = time.time() + 2

                if is_static_object:
                    continue
                
                # Stabilization delay for logic processing (still show skeleton above)
                if not person_is_confirmed.get(persistent_id, False) and person_frames_seen[persistent_id] < 10:
                    continue

                # 4. Resolve Display Name (Face/Manual Name > Persistent ID)
                pid = get_display_id(persistent_id)
                
                # 4. Periodically try to "Name" the Persistent ID using Face Recognition
                if FACE_RECOGNITION_AVAILABLE and not SINGLE_PERSON_MODE and frame_count % 60 == 0 and pid == persistent_id: # Only if not already named
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        with data_lock:
                            target_encodings = known_face_encodings[:]
                            target_names = known_face_names[:]
                        
                        if target_encodings:
                            rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_person, number_of_times_to_upsample=1)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                                for fe in face_encodings:
                                    matches = face_recognition.compare_faces(target_encodings, fe, tolerance=0.6)
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        real_name = str(target_names[first_match_index])
                                        rename_person(persistent_id, real_name)
                                        pid = real_name
                                        break

                primary_target_id = get_primary_monitored_id()
                if primary_target_id is None:
                    monitored_persistent_id = str(persistent_id)
                    primary_target_id = str(persistent_id)
                elif ward_profile.get("name") and manual_id_map.get(str(persistent_id)) == ward_profile.get("name"):
                    monitored_persistent_id = str(persistent_id)
                    primary_target_id = str(persistent_id)
                    ward_locked_persistent_id = str(persistent_id)
                elif SINGLE_PERSON_MODE and ward_profile.get("name") and visible_people_count == 1:
                    monitored_persistent_id = str(persistent_id)
                    primary_target_id = str(persistent_id)
                    ward_locked_persistent_id = str(persistent_id)
                    manual_id_map[str(persistent_id)] = ward_profile.get("name", SINGLE_PERSON_LABEL)
                    save_manual_id_map()
                is_guest_detection = (not multi_person_visible) and str(persistent_id) != str(primary_target_id)

                if is_guest_detection:
                    guest_now = time.time()
                    if persistent_id not in guest_state:
                        guest_state[persistent_id] = "UNKNOWN"
                        guest_last_time[persistent_id] = guest_now
                        guest_first_seen[persistent_id] = guest_now
                    guest_last_seen[persistent_id] = guest_now

                    guest_prev_state = guest_state.get(persistent_id, "UNKNOWN")
                    guest_duration = guest_now - guest_last_time.get(persistent_id, guest_now)
                    if guest_duration > 0:
                        if guest_prev_state == "WALKING":
                            guest_walking_time[persistent_id] += guest_duration
                        elif guest_prev_state == "STANDING":
                            guest_standing_time[persistent_id] += guest_duration
                        elif guest_prev_state == "SITTING":
                            guest_sitting_time[persistent_id] += guest_duration
                        elif guest_prev_state == "SLEEPING":
                            guest_sleeping_time[persistent_id] += guest_duration
                    guest_last_time[persistent_id] = guest_now
                    if activity != "UNKNOWN":
                        guest_state[persistent_id] = activity

                    if settings.get("display_metrics_overlay", True):
                        guest_name = get_guest_display_name(persistent_id)
                        guest_activity = guest_state.get(persistent_id, "UNKNOWN")
                        cv2.putText(
                            display_frame,
                            f"{guest_name}: {guest_activity} | Conf {pose_confidence:.2f}",
                            (20, 60 + (i % 5) * 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            (210, 210, 210),
                            2
                        )
                    continue
                
                # Update tracking metadata
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                
                # 5. Classify Activity
                keypoints = kp.cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                now = time.time()
                
                # Calculate velocity (rolling average displacement)
                if persistent_id in person_last_pos:
                    last_pos = person_last_pos[persistent_id]
                    dist = np.sqrt((center_coords[0]-last_pos[0])**2 + (center_coords[1]-last_pos[1])**2)
                    v_dist = center_coords[1] - last_pos[1] # Positive is downward
                    person_velocity[persistent_id] = person_velocity[persistent_id] * 0.8 + dist * 0.2
                    person_vertical_velocity[persistent_id] = person_vertical_velocity[persistent_id] * 0.8 + v_dist * 0.2
                
                person_last_pos[persistent_id] = center_coords
                current_velocity = person_velocity[persistent_id]
                current_v_velocity = person_vertical_velocity[persistent_id]
                torso_angle, shoulder_center, hip_center = get_torso_angle(keypoints, confidences)
                angle_rate = 0.0
                if torso_angle is not None and persistent_id in person_previous_torso_angle and persistent_id in person_previous_torso_time:
                    dt_angle = max(1e-3, now - person_previous_torso_time[persistent_id])
                    angle_rate = abs(torso_angle - person_previous_torso_angle[persistent_id]) / dt_angle
                if torso_angle is not None:
                    person_previous_torso_angle[persistent_id] = torso_angle
                    person_previous_torso_time[persistent_id] = now
                
                # Aspect Ratio of bounding box (width/height)
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                aspect_ratio = bw / bh if bh > 0 else 0
                body_spread_ratio = get_body_spread_ratio(keypoints, confidences)
                normalized_current_velocity = normalize_motion_for_fps(current_velocity, fps_hint=MOTION_REFERENCE_FPS)
                normalized_current_v_velocity = normalize_motion_for_fps(current_v_velocity, fps_hint=MOTION_REFERENCE_FPS)
                recent_sudden_drop = False
                if normalized_current_v_velocity > SUDDEN_DROP_VERTICAL_SPEED or normalized_current_velocity > SUDDEN_DROP_MOTION_SCORE:
                    person_recent_sudden_drop_until[persistent_id] = now + SUDDEN_DROP_HOLD_SECONDS
                if now < person_recent_sudden_drop_until.get(persistent_id, 0.0):
                    recent_sudden_drop = True

                horizontal_candidate = (
                    torso_angle is not None
                    and (torso_angle >= LYING_TRANSITION_ANGLE_MIN or aspect_ratio >= LYING_TRANSITION_BOX_RATIO)
                )
                if horizontal_candidate:
                    if persistent_id not in person_transition_start:
                        person_transition_start[persistent_id] = now
                elif persistent_id in person_transition_start:
                    del person_transition_start[persistent_id]

                stable_horizontal_candidate = (
                    torso_angle is not None
                    and torso_angle >= LYING_HORIZONTAL_ANGLE_MIN
                    and aspect_ratio >= LYING_HORIZONTAL_BOX_RATIO
                    and normalized_current_velocity <= LYING_MAX_MOTION_SCORE
                    and angle_rate <= LYING_MAX_ANGLE_RATE
                    and not recent_sudden_drop
                )
                if stable_horizontal_candidate:
                    if persistent_id not in person_horizontal_stable_start:
                        person_horizontal_stable_start[persistent_id] = now
                elif persistent_id in person_horizontal_stable_start:
                    del person_horizontal_stable_start[persistent_id]

                stable_horizontal_time = max(0.0, now - person_horizontal_stable_start[persistent_id]) if persistent_id in person_horizontal_stable_start else 0.0
                transition_time = max(0.0, now - person_transition_start[persistent_id]) if persistent_id in person_transition_start else 0.0
                
                activity = classify_activity(
                    keypoints,
                    confidences,
                    velocity=current_velocity,
                    v_velocity=current_v_velocity,
                    aspect_ratio=aspect_ratio,
                    fps_hint=MOTION_REFERENCE_FPS,
                    angle_rate=angle_rate,
                    stable_horizontal_time=stable_horizontal_time,
                    transition_time=transition_time,
                    recent_sudden_drop=recent_sudden_drop,
                    body_spread_ratio=body_spread_ratio
                )

                # --- 6. Body Scanning (Capture multi-angle signatures) ---
                # During the first 10 seconds of seeing a person, periodically capture different angles
                with data_lock:
                    id_data = reid_manager.identity_bank.get(persistent_id, {})
                    first_seen = id_data.get('first_seen', 0)
                    gallery_len = len(id_data.get('embeddings', [])) if 'embeddings' in id_data else 1
                
                if IDENTITY_MODE == "reid" and (now - first_seen) < 15 and frame_count % 15 == 0:
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        # Periodically update color signature to handle lighting changes
                        current_sig = get_color_signature(person_img)
                        emb = reid_manager.get_embedding(person_img)
                        
                        with data_lock:
                            if persistent_id in reid_manager.identity_bank:
                                # Update color signature
                                reid_manager.identity_bank[persistent_id]['color_sig'] = current_sig
                                # Update embedding via moving average if already matched
                                old_emb = reid_manager.identity_bank[persistent_id].get('embedding')
                                if old_emb is not None:
                                    updated_emb = 0.9 * old_emb + 0.1 * emb # Slower update during tracking
                                    norm = np.linalg.norm(updated_emb)
                                    if norm > 0:
                                        reid_manager.identity_bank[persistent_id]['embedding'] = updated_emb / norm
                                else:
                                    # Fallback for legacy gallery if needed
                                    reid_manager.add_to_gallery(persistent_id, emb)
                        
                        # Visual feedback for scanning
                        cv2.putText(display_frame, "Updating Body Signature...", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                with data_lock:
                    if persistent_id not in person_state:
                        person_state[persistent_id] = "UNKNOWN"
                        person_last_time[persistent_id] = now
                        all_tracked_people.add(pid) # Store resolved name for DB/Reporting
                        print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")

                # State Machine logic for ESCALATION and RECOVERY
                new_state = activity
                if activity == "UNKNOWN":
                    new_state = person_state.get(persistent_id, "UNKNOWN")
                
                # --- State Transition Refinements ---
                prev_s = person_state.get(persistent_id, "UNKNOWN")
                
                # If already in a confirmed MAJOR FALL, stay there until recovery
                if prev_s == "MAJOR FALL" and new_state in ["MINOR FALL", "LYING"]:
                    new_state = "MAJOR FALL"

                # SUPPRESS "getting up" misclassification:
                # If they are moving UP (negative v_velocity) and were previously down, 
                # they are likely getting up. Force upright state to prevent false Minor Fall.
                if normalized_current_v_velocity < -1.0 and prev_s in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]:
                    if new_state in ["MINOR FALL", "LYING"]:
                        new_state = "STANDING" 
                if prev_s in ["LYING", "SLEEPING", "TRANSITION"] and new_state == "MINOR FALL":
                    new_state = "TRANSITION"
                if prev_s in ["TRANSITION", "LYING", "SLEEPING"] and new_state in ["STANDING", "WALKING"]:
                    if normalized_current_v_velocity > -1.2 and normalized_current_velocity <= LYING_MAX_MOTION_SCORE:
                        new_state = prev_s
                if prev_s == "TRANSITION" and transition_time >= DOWN_STATE_HOLD_SECONDS and normalized_current_velocity <= LYING_MAX_MOTION_SCORE:
                    new_state = "LYING"
                if prev_s in ["TRANSITION", "LYING", "SLEEPING"] and normalized_current_velocity <= LYING_MAX_MOTION_SCORE:
                    if transition_time >= SLEEPING_AFTER_LYING_SECONDS:
                        new_state = "SLEEPING"
                    elif transition_time >= DOWN_STATE_HOLD_SECONDS and new_state in ["TRANSITION", "STANDING", "WALKING"]:
                        new_state = "LYING"

                # 4. Special case: If in recovery mode, show RECOVERED label briefly
                if persistent_id in recovery_mode:
                    if now > recovery_mode[persistent_id]: 
                        del recovery_mode[persistent_id]
                    else:
                        # If they briefly tilt while getting up/stabilizing, don't trigger a new fall
                        if new_state in ["MINOR FALL", "LYING"]:
                            new_state = "STANDING" # Keep them upright during stabilization
                        
                        # Show RECOVERED label for 5 seconds (of the 10s recovery window)
                        if now < (recovery_mode[persistent_id] - 5.0):
                            new_state = "RECOVERED"

                # 1. Handle Risk States (Lying or Minor Fall)
                is_currently_down = (new_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"])
                
                # INITIAL FALL DETECTION
                if is_currently_down and persistent_id not in active_fall_event:
                    # Trigger alert immediately if they transitioned from WALKING/STANDING
                    # Exclude SITTING: Sitting -> Lying/Minor Fall is considered intentional/sleeping
                    if prev_s in ["WALKING", "STANDING", "RECOVERED"]:
                        if new_state in ["LYING", "SLEEPING"]:
                            new_state = "MINOR FALL"
                        active_fall_event[persistent_id] = "MINOR"
                        send_fall_alert(f"MINOR FALL (ID {pid})", pid, "MINOR FALL", coords=center_coords)
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MINOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })

                if is_currently_down:
                    recovery_confirm_count[persistent_id] = 0 # Reset recovery counter
                    if persistent_id not in minor_fall_start_time:
                        # Start timer to track duration for MAJOR FALL escalation
                        # Only start if transitioning from an upright state
                        if prev_s in ["WALKING", "STANDING", "RECOVERED"]:
                            minor_fall_start_time[persistent_id] = now
                    
                    # Escalation check: 10 seconds after being down (Only if it was a fall, not sleep)
                    # USER REQUEST: Only if minor fall AND lying down for 10s
                    confirm_window = float(settings.get("fall_confirm_window_sec", 10.0))
                    if active_fall_event.get(persistent_id) == "MINOR" and new_state in ["LYING", "MAJOR FALL"] and (now - minor_fall_start_time.get(persistent_id, now) > confirm_window):
                        send_fall_alert(f"MAJOR FALL (ID {pid}) - No recovery after {confirm_window:.1f}s", pid, "MAJOR FALL", coords=center_coords)
                        active_fall_event[persistent_id] = "MAJOR"
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MAJOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })
                
                # 2. Handle Potential Recovery (Upright: WALKING, STANDING, SITTING)
                elif new_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"]:
                    # Require 30 frames (~1s at 30fps) of consistent upright pose before confirming recovery
                    recovery_confirm_count[persistent_id] = recovery_confirm_count.get(persistent_id, 0) + 1
                    
                    if recovery_confirm_count[persistent_id] > 30:
                        if persistent_id in active_fall_event:
                            recovery_mode[persistent_id] = now + 10.0 # 10s stabilization window
                            send_fall_alert(f"RECOVERED (ID {pid})", pid, "RECOVERED", coords=center_coords)
                            with data_lock:
                                fall_events.append({
                                    "person": pid, "type": "RECOVERED", "timestamp": now,
                                    "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                                })
                            del active_fall_event[persistent_id]
                        
                        # Always clear "down" timers if confirmed upright
                        if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                        if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                        recovery_confirm_count[persistent_id] = 0

                # 3. SLEEPING logic (sustained lying or sitting-to-lying)
                if new_state == "LYING":
                    if persistent_id not in lying_start_time: 
                        lying_start_time[persistent_id] = now
                    
                    # USER REQUEST: Sitting -> Lying is considered sleeping/intentional
                    if prev_s in ["SITTING", "SLEEPING"] or (prev_s == "UNKNOWN" and normalized_current_v_velocity < 4.5):
                        new_state = "SLEEPING"
                    elif now - lying_start_time[persistent_id] > SLEEPING_AFTER_LYING_SECONDS:
                        new_state = "SLEEPING"
                elif new_state != "SLEEPING":
                    # Clear lying timer if they are not lying or already sleeping
                    if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                
                # Accumulate time for CURRENT activity only in single-person scenes.
                duration = now - person_last_time[persistent_id]
                if duration > 0 and not multi_person_visible:
                    with data_lock:
                        prev_state = person_state.get(persistent_id, "UNKNOWN")
                        if prev_state == "WALKING": walking_time[pid] += duration
                        elif prev_state == "STANDING": standing_time[pid] += duration
                        elif prev_state == "SITTING": sitting_time[pid] += duration
                        elif prev_state == "SLEEPING": sleeping_time[pid] += duration
                    
                person_last_time[persistent_id] = now

                # Update state if changed
                if new_state != "UNKNOWN" and new_state != prev_s:
                    with data_lock:
                        person_state[persistent_id] = new_state
                    if (not multi_person_visible) or ("FALL" in new_state) or (new_state == "RECOVERED"):
                        notify_activity_change(pid, new_state)

                # Overlay text
                if settings.get("display_metrics_overlay", True):
                    with data_lock:
                        walk_str = format_duration(walking_time[pid])
                        stand_str = format_duration(standing_time[pid])
                        sit_str = format_duration(sitting_time[pid])
                        sleep_str = format_duration(sleeping_time[pid])
                        
                        # Color code based on state
                        current_s = person_state.get(persistent_id, "UNKNOWN")
                        if "FALL" in current_s:
                            color = (0, 0, 255) # Red for fall
                        elif current_s == "RECOVERED":
                            color = (0, 255, 0) # Green for recovery
                        elif current_s == "WALKING":
                            color = (0, 255, 255) # Yellow for walking
                        elif current_s == "STANDING":
                            color = (255, 200, 0) # Orange for standing
                        else:
                            color = (255, 0, 0) # Blue for sitting/sleeping/unknown
                        
                        overlay_state = current_s
                        cv2.putText(display_frame, f"ID {pid}: {overlay_state} | Conf {pose_confidence:.2f}", (20, 60 + (i % 5) * 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        if multi_person_visible:
                            cv2.putText(display_frame, "Multiple people detected - normal activity tracking paused", (20, 90 + (i % 5) * 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
                        else:
                            cv2.putText(display_frame, f"W:{walk_str} St:{stand_str} S:{sit_str} Sl:{sleep_str}", (20, 90 + (i % 5) * 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No keypoints detected
            if settings.get("display_metrics_overlay", True):
                cv2.putText(display_frame, "No pose detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if pending_registration.get("active") and not ward_registration_session.get("active"):
            preferred_target = str(pending_registration.get("preferred_target_id") or "")
            full_body_candidates = [
                pid for pid in detected_ids
                if latest_person_pose_meta.get(str(pid), {}).get("full_body_visible")
            ]
            chosen_candidate = None
            if preferred_target and preferred_target in full_body_candidates:
                chosen_candidate = preferred_target
            elif len(full_body_candidates) == 1:
                chosen_candidate = full_body_candidates[0]

            if chosen_candidate is not None:
                rename_person(str(chosen_candidate), pending_registration.get("name", SINGLE_PERSON_LABEL))
                ward_locked_persistent_id = str(chosen_candidate)
                start_ward_registration_session(str(chosen_candidate), pending_registration.get("name", SINGLE_PERSON_LABEL))
                status_message = f"Full body detected for {pending_registration.get('name', SINGLE_PERSON_LABEL)}. Automatic 360 capture starts in 5 seconds."
                status_expiry = time.time() + 6
            elif settings.get("display_metrics_overlay", True):
                cv2.putText(
                    display_frame,
                    "Registration queued: step back until one full body is visible",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2
                )

        if person_visible_this_frame:
            out_of_frame_since = None
        else:
            if now < wake_grace_until:
                if settings.get("display_metrics_overlay", True):
                    remaining_grace = max(0.0, wake_grace_until - now)
                    cv2.putText(
                        display_frame,
                        f"WAKING UP - HOLD STILL {remaining_grace:.1f}s",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 220, 255),
                        2
                    )
            else:
                if out_of_frame_since is None:
                    out_of_frame_since = now
                elif (now - out_of_frame_since) >= LOW_POWER_IDLE_TIMEOUT_SEC:
                    enter_low_power_mode(frame, w)
                    if cap:
                        cap.release()
                    cap = None
                    continue

                if settings.get("display_metrics_overlay", True):
                    elapsed_out = now - out_of_frame_since
                    remaining_out = max(0.0, LOW_POWER_IDLE_TIMEOUT_SEC - elapsed_out)
                    cv2.putText(
                        display_frame,
                        f"OUT OF FRAME - SLEEP IN {remaining_out:.1f}s",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2
                    )

        # Clean up people not detected for 30 frames (about 1 second at 30fps)
        ids_to_remove = []
        for persistent_id in list(person_state.keys()):
            # Stability Patch: Don't remove named/important IDs from active tracking state easily
            if persistent_id not in detected_ids and (frame_count - last_detection.get(persistent_id, 0)) > 30:
                if persistent_id in manual_id_map:
                    # Named IDs get a much longer timeout (e.g. 5 minutes) before being cleared from memory
                    if (frame_count - last_detection.get(persistent_id, 0)) > 9000:
                         ids_to_remove.append(persistent_id)
                else:
                    ids_to_remove.append(persistent_id)
        guest_ids_to_remove = [
            persistent_id for persistent_id in list(guest_state.keys())
            if persistent_id not in detected_ids and (frame_count - last_detection.get(persistent_id, 0)) > 30
        ]
        
        with data_lock:
            for persistent_id in ids_to_remove:
                if persistent_id in person_state: del person_state[persistent_id]
                if persistent_id in person_last_time: del person_last_time[persistent_id]
                if persistent_id in last_detection: del last_detection[persistent_id]
                if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                if persistent_id in recovery_mode: del recovery_mode[persistent_id]
                if persistent_id in active_fall_event: del active_fall_event[persistent_id]
                if persistent_id in recovery_confirm_count: del recovery_confirm_count[persistent_id]
                if persistent_id in person_start_pos: del person_start_pos[persistent_id]
                if persistent_id in person_frames_seen: del person_frames_seen[persistent_id]
                if persistent_id in person_is_confirmed: del person_is_confirmed[persistent_id]
                if persistent_id in person_velocity: del person_velocity[persistent_id]
                if persistent_id in person_last_pos: del person_last_pos[persistent_id]
                if persistent_id in person_previous_torso_angle: del person_previous_torso_angle[persistent_id]
                if persistent_id in person_previous_torso_time: del person_previous_torso_time[persistent_id]
                if persistent_id in person_horizontal_stable_start: del person_horizontal_stable_start[persistent_id]
                if persistent_id in person_transition_start: del person_transition_start[persistent_id]
                if persistent_id in person_recent_sudden_drop_until: del person_recent_sudden_drop_until[persistent_id]
                
                # Clean up tracker mapping to prevent stale entries
                yolo_keys = [k for k, v in tracker_to_persistent.items() if v == persistent_id]
                for k in yolo_keys: del tracker_to_persistent[k]
                
                print(f"Removed internal ID {persistent_id} from active tracking")
                if str(monitored_persistent_id) == str(persistent_id) and not ward_locked_persistent_id:
                    monitored_persistent_id = None

            for persistent_id in guest_ids_to_remove:
                guest_state.pop(persistent_id, None)
                guest_last_time.pop(persistent_id, None)
                guest_first_seen.pop(persistent_id, None)
                guest_last_seen.pop(persistent_id, None)
                guest_walking_time.pop(persistent_id, None)
                guest_standing_time.pop(persistent_id, None)
                guest_sitting_time.pop(persistent_id, None)
                guest_sleeping_time.pop(persistent_id, None)

        # Log progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            update_evaluation_metric("fps", round(fps, 1))
            update_evaluation_metric("people_tracked", len(person_state))
            print(f"Frame {frame_count} | FPS: {fps:.1f} | People tracked: {len(person_state)}")

        draw_ward_registration_overlay(display_frame)

        # Show status message if active
        if time.time() < status_expiry:
            status_bar_y = 110 if ward_registration_session.get("active") else 0
            cv2.rectangle(display_frame, (0, status_bar_y), (w, status_bar_y + 40), (0, 255, 0), -1)
            cv2.putText(display_frame, status_message, (10, status_bar_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        show_preview_window(display_frame)
        update_stream_frame(display_frame)

        if is_edge_mode() and (time.time() - last_edge_sync_at) >= 2:
            send_edge_report_snapshot()
            last_edge_sync_at = time.time()

    except Exception as e:
        print(f"Error at frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        break

try:
    if cap is not None:
        cap.release()
except Exception:
    pass

try:
    cv2.destroyAllWindows()
except Exception:
    pass

print("Program exited.")
sys.exit(0)
