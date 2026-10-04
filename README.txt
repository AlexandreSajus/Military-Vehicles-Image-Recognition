UNIFIED SAFETY VISION - LOCAL WEBCAM DEMO

This small package detects people locally from a webcam or video and raises a
few transparent signals for a person to review. It is not an identity,
emotion, intent, criminality, or danger detector. It does not recognize faces
or save images/video. Events are advisory; no machine or emergency system is
controlled by this demo.

START ON WINDOWS

1. Extract this ZIP to a folder.
2. Install Python 3.9 or newer if it is not already installed.
3. Double-click START_DEMO.bat.
4. Allow webcam access if Windows asks.
5. Press H in the video window to record a manual help request. Press Q or Esc
   to quit.

The first run creates a local .venv, installs the listed packages, and
downloads yolov8n.pt from Ultralytics if the model is not already cached.
Internet access is required for those first-run downloads. The model can run
on CPU; CUDA acceleration is only used when a compatible PyTorch/CUDA setup is
available. Accelerator support and speed depend on the computer and installed
PyTorch build.

COMMAND LINE

From this folder:
  python -m pip install -r requirements.txt
  python human_webcam_demo.py --source 0 --weights yolov8n.pt --imgsz 480

Use a video file instead of a webcam:
  python human_webcam_demo.py --source "C:\path\to\video.mp4"

Optional industrial-area presence alert (rectangle coordinates are normalized
to the video width/height, from 0 to 1):
  python human_webcam_demo.py --source 0 --zone 0.35,0.15,0.75,0.95

Optional prolonged-presence review threshold:
  python human_webcam_demo.py --source 0 --prolonged-seconds 45

WHAT THE SIGNALS MEAN

- Configured zone entry: a detected person is inside the user-drawn rectangle.
  It produces a HIGH advisory event for the site controller to verify and
  follow local procedures. It does not detect equipment faults or stop a
  process.
- Prolonged presence: a person remains visible for the configured time. It
  produces a REVIEW event only; presence duration does not imply wrongdoing or
  danger.
- Manual help request: pressing H writes an URGENT local event so an operator
  can check on the person. The program cannot determine whether somebody is
  uneasy or needs help without an explicit request.

LOCAL OUTPUT

Events are written beside the script in runs\safety_demo:
- event_log.csv
- volunteer_alerts.jsonl

Only event time, event category, advisory text, and temporary detector track
reference are logged. Track references are not identities and may change
between runs. No frames, crops, face templates, or location coordinates are
saved. Restrict access to the output folder and delete the logs when they are
no longer needed.

LIMITATIONS

The pretrained model is a general person detector, not a validated security
or industrial safety system. Lighting, camera placement, occlusion, and model
errors can cause missed or false detections. Zone alerts work only for the
rectangle configured by the operator. Product defect/quality inspection,
equipment-fault detection, formal regulatory compliance, person
re-identification, and automatic emergency actions are not implemented.
Test the demo in a controlled setting and keep a human responsible for every
decision.
