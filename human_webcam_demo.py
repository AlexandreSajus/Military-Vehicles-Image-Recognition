"""Local webcam demo for person detection and simple, review-only safety alerts."""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger("unified_safety_demo")


def parse_zone(value):
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("zone must be x1,y1,x2,y2") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("zone must be x1,y1,x2,y2")
    x1, y1, x2, y2 = coordinates
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise argparse.ArgumentTypeError(
            "zone coordinates must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
    return coordinates


def parse_source(value):
    return int(value) if value.isdecimal() else value


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLogger:
    """Append low-data event records locally; never save frames or face data."""

    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.csv_file = (directory / "event_log.csv").open(
            "a", newline="", encoding="utf-8"
        )
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(
                ["timestamp_utc", "event", "priority", "track_ref", "message"]
            )
            self.csv_file.flush()
        self.alert_file = (directory / "volunteer_alerts.jsonl").open(
            "a", encoding="utf-8"
        )

    def write(self, event, priority, track_ref, message):
        moment = timestamp()
        self.csv_writer.writerow([moment, event, priority, track_ref or "", message])
        self.csv_file.flush()
        if priority in {"HIGH", "URGENT"} or event == "manual_help_request":
            self.alert_file.write(
                json.dumps(
                    {
                        "timestamp_utc": moment,
                        "event": event,
                        "priority": priority,
                        "track_ref": track_ref,
                        "message": message,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            self.alert_file.flush()

    def close(self):
        self.csv_file.close()
        self.alert_file.close()


class SafetyMonitor:
    """Apply transparent presence/zone rules to temporary tracker IDs."""

    def __init__(self, zone, prolonged_seconds, logger):
        self.zone = zone
        self.prolonged_seconds = prolonged_seconds
        self.logger = logger
        self.tracks = {}

    def update(self, centers, frame_width, frame_height, now):
        visible = set()
        statuses = {}
        for track_ref, center_x, center_y in centers:
            if track_ref is None:
                continue
            visible.add(track_ref)
            state = self.tracks.setdefault(
                track_ref,
                {
                    "started": now,
                    "last_seen": now,
                    "in_zone": False,
                    "prolonged_logged": False,
                },
            )
            state["last_seen"] = now
            status = "PERSON"
            if self.zone is not None:
                x1, y1, x2, y2 = self.zone
                fx, fy = center_x / frame_width, center_y / frame_height
                in_zone = x1 <= fx <= x2 and y1 <= fy <= y2
                if in_zone and not state["in_zone"]:
                    self.logger.write(
                        "restricted_zone_entry",
                        "HIGH",
                        track_ref,
                        "Person detected in configured zone. Notify the site controller "
                        "and follow local stop/check procedures.",
                    )
                state["in_zone"] = in_zone
                if in_zone:
                    status = "ZONE ENTRY"
            if (
                not state["prolonged_logged"]
                and now - state["started"] >= self.prolonged_seconds
            ):
                state["prolonged_logged"] = True
                self.logger.write(
                    "prolonged_presence",
                    "REVIEW",
                    track_ref,
                    "Person remained visible for the configured duration; "
                    "request a human review if appropriate.",
                )
                if status == "PERSON":
                    status = "REVIEW"
            statuses[track_ref] = status

        for track_ref in tuple(self.tracks):
            if track_ref not in visible and now - self.tracks[track_ref]["last_seen"] > 2:
                del self.tracks[track_ref]
        return statuses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local person-detection demo with configurable, human-reviewed safety alerts."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="webcam index (default: 0) or a video file path",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Ultralytics model weights (default: yolov8n.pt; downloaded on first use)",
    )
    parser.add_argument("--imgsz", type=int, default=480, help="inference size")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a supported accelerator identifier (default: auto)",
    )
    parser.add_argument(
        "--zone",
        type=parse_zone,
        help="optional normalized rectangle x1,y1,x2,y2 (e.g. 0.35,0.15,0.75,0.95)",
    )
    parser.add_argument(
        "--prolonged-seconds",
        type=float,
        default=30.0,
        help="seconds in view before a review-only prolonged-presence event (default: 30)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent / "runs" / "safety_demo"),
        help="directory for local event logs",
    )
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    if args.prolonged_seconds <= 0:
        parser.error("--prolonged-seconds must be positive")
    args.source = parse_source(args.source)
    return args


def choose_device(requested):
    if requested.lower() != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def run(args):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Required packages are missing. Run START_DEMO.bat or "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    device = choose_device(args.device)
    LOGGER.info("Loading weights: %s", args.weights)
    LOGGER.info("Inference device: %s (accelerator support depends on this computer)", device)
    model = YOLO(args.weights)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {args.source}")

    event_logger = EventLogger(args.log_dir)
    monitor = SafetyMonitor(args.zone, args.prolonged_seconds, event_logger)
    frame_count = 0
    had_frames = False
    fps = 0.0
    fps_started = time.monotonic()
    inference_ms = 0.0
    last_statuses = {}
    LOGGER.info("Press H to record a manual help request; press Q or Esc to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if not had_frames:
                    raise RuntimeError("The video source opened but returned no frames.")
                break
            had_frames = True

            started = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - started) * 1000
            result = results[0]
            centers = []
            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                coordinates = boxes.xyxy.cpu().tolist()
                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(coordinates)
                )
                confidences = boxes.conf.cpu().tolist()
                for xyxy, track_ref, confidence in zip(
                    coordinates, track_ids, confidences
                ):
                    x1, y1, x2, y2 = (int(value) for value in xyxy)
                    footpoint = ((x1 + x2) / 2, y2)
                    centers.append((track_ref, *footpoint))
                    detections.append((x1, y1, x2, y2, track_ref, confidence))

            height, width = frame.shape[:2]
            last_statuses = monitor.update(centers, width, height, time.monotonic())
            for x1, y1, x2, y2, track_ref, confidence in detections:
                status = last_statuses.get(track_ref, "PERSON")
                color = (0, 0, 255) if status == "ZONE ENTRY" else (0, 200, 255)
                label = f"Person {track_ref if track_ref is not None else ''} {confidence:.2f} {status}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label.strip(),
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if args.zone is not None:
                x1, y1, x2, y2 = args.zone
                left, top = int(x1 * width), int(y1 * height)
                right, bottom = int(x2 * width), int(y2 * height)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Configured review zone",
                    (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            frame_count += 1
            elapsed = time.monotonic() - fps_started
            if elapsed >= 1:
                fps = frame_count / elapsed
                fps_started = time.monotonic()
                frame_count = 0
            active_count = len(last_statuses)
            cv2.putText(
                frame,
                f"Device: {device} | FPS: {fps:.1f} | Inference: {inference_ms:.0f} ms | People: {active_count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "H: manual help request | Q/Esc: quit",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Unified Safety Vision - local review demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                event_logger.write(
                    "manual_help_request",
                    "URGENT",
                    None,
                    "A local operator pressed H; check on the person and follow site procedures.",
                )
                LOGGER.warning("Manual help request recorded in the local alert log.")
            elif key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        event_logger.close()
    LOGGER.info("Local event logs: %s", Path(args.log_dir).resolve())
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args())
    except (RuntimeError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
