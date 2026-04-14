Elderly Fall Detection and Activity Monitoring System

This project is a real-time intelligent monitoring system designed for elderly care scenarios. It combines pose estimation, activity understanding, fall detection, identity persistence, web-based monitoring, alerting, and low-power behavior into a single application.

The system is built around a pretrained pose estimation model and custom decision logic. It can classify daily activities, detect fall events, show a live dashboard, send alerts, and support a small distributed setup where one laptop acts as the server and other laptops act as edge monitoring nodes.

What the project does

- Detects human pose in real time using `YOLO11n-pose`
- Classifies posture and movement into:
  - `WALKING`
  - `STANDING`
  - `SITTING`
  - `SLEEPING`
  - `LYING`
  - `MINOR FALL`
  - `MAJOR FALL`
  - `RECOVERED`
- Escalates a minor fall into a major fall if recovery does not happen within the configured confirmation window
- Tracks activity durations over time for walking, standing, sitting, and sleeping
- Stores activity summaries and fall history in SQLite
- Supports optional face registration and person naming
- Supports person re-identification across sessions using ReID and appearance cues
- Sends Telegram alerts
- Runs a modern web dashboard with live updates
- Enters low-power mode when the monitored person leaves the frame
- Wakes up again when motion is detected
- Supports server and edge deployment modes for multi-laptop monitoring

How the AI part works

This project is not a single custom-trained end-to-end deep learning model. It is a hybrid intelligent system.

The AI-related components are:

- A pretrained `YOLO11n-pose` model for pose estimation
- Optional pretrained face recognition for naming
- Optional pretrained person ReID for identity persistence

The final fall detection and activity classification logic is custom system logic written in Python. It uses:

- body keypoint geometry
- body angle
- aspect ratio
- movement speed
- vertical motion
- temporal rules
- recovery timers
- escalation windows

In other words, the base pose model is pretrained, while the fall and activity reasoning layer is implemented as rule-based inference on top of those pose outputs.

Main features

Live monitoring

- A processed live video feed is shown in the browser dashboard
- An OpenCV preview window can also be shown locally
- Pose overlays include body joints and enhanced hand indicators
- Current person state and activity durations are drawn on the frame

Fall detection

- Detects immediate risk states based on pose and motion
- Generates `MINOR FALL` alerts first
- Escalates to `MAJOR FALL` if the person remains down after the configured confirmation window
- Detects `RECOVERED` when the person returns to a stable upright state
- Stores fall history in the local database

Identity and person tracking

- Uses YOLO track IDs in tracker mode
- Uses ReID embeddings when available
- Supports manual naming from the dashboard
- Can load and save persistent identity banks between sessions

Daily wellness tracking

- Tracks total time spent walking, standing, sitting, and sleeping
- Shows daily summary cards
- Generates recommendations based on activity balance
- Includes caregiver insight messages with `Stable`, `Watch`, or `Urgent` status

Low-power behavior

- Detects when the monitored person leaves the frame
- Shows a low-power notice on both the dashboard stream and preview window
- Stops active camera use after the timeout
- Reopens when motion is detected again
- Tracks wake-up behavior in evaluation metrics

Dashboard

The dashboard includes:

- live video feed
- active fall alerts
- daily summary
- caregiver insights
- evaluation metrics
- managed people cards
- historical fall feed
- daily and monthly analytics charts
- connected node visibility for distributed monitoring

Multi-laptop deployment

The system supports:

- `server` mode
- `edge` mode
- `standalone` mode

Typical use:

- one laptop runs as the central dashboard server
- one or more laptops run as edge devices with their own cameras
- edge nodes send heartbeats and report snapshots to the server
- the server dashboard shows connected nodes and merges remote reports

Project files

- `smart_fall_activity_report.py`
  Main application file. It contains the Flask dashboard, camera loop, pose processing, fall logic, database logic, ReID logic, Telegram integration, and edge/server networking.

- `requirements.txt`
  Python dependency list for the project.

- `yolo11n-pose.pt`
  Pose estimation model weights used by the application.

- `system_settings.json`
  Runtime configuration file for Telegram, deployment mode, camera selection, and dashboard behavior.

- `monitor_data.db`
  SQLite database used to store activity summaries and fall history. This file is created and updated automatically.

- `registered_faces/`
  Folder used for saved face registration data.

- `encodings.pickle`
  Stored face encodings used for person naming.

- `reid_bank.pickle`
  Persistent ReID identity bank.

- `manual_id_map.pickle`
  Manual name mapping storage.

Requirements

- Windows laptop or desktop recommended
- Python 3.11 or newer recommended
- Webcam or integrated camera
- Internet connection for first-time dependency and model setup
- NVIDIA GPU recommended for smooth real-time performance

The project can run on CPU, but GPU is strongly recommended for better responsiveness.

Verified GPU setup in this project

This setup has already been prepared to run with CUDA when available. In the current environment, the project was configured to use:

- `torch 2.9.1+cu128`
- `cuda:0`
- `NVIDIA GeForce RTX 3050 A Laptop GPU`

Installation

1. Open a terminal inside the project folder.

2. Create a virtual environment if needed:

```powershell
python -m venv .venv
```

3. Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Make sure the model file exists:

- `yolo11n-pose.pt`

If it is missing, the application may try to download it automatically, or you can place it manually in the project folder.

Running the project

Run the application with:

```powershell
python smart_fall_activity_report.py
```

When the application starts:

- Flask server starts in the background
- the dashboard becomes available in the browser
- the camera loop starts
- low-power mode and motion wake logic are active

Open the dashboard at:

- `http://127.0.0.1:5000/`

Settings page:

- `http://127.0.0.1:5000/settings`

Deployment modes

Server mode

Use this when the laptop should host the dashboard and receive updates from edge nodes.

Recommended settings:

- `Deployment Mode` = `server`
- `Server Bind Host` = `0.0.0.0`
- `Server Port` = `5000`
- `Node ID` = a clear name such as `main-laptop`

Edge mode

Use this when the laptop should monitor locally and report to a server laptop.

Recommended settings:

- `Deployment Mode` = `edge`
- `Central Server URL` = `http://SERVER_IP:5000`
- `Node ID` = a clear unique name such as `room-2-laptop`

Standalone mode

Use this when the laptop should run alone without remote edge synchronization.

Recommended settings:

- `Deployment Mode` = `standalone`

Multi-node setup example

Main laptop:

- connected to network
- runs in `server` mode
- dashboard is opened on this machine

Friend laptop or room laptop:

- runs the same project
- set to `edge` mode
- points `Central Server URL` to the server laptop IP

If using Windows mobile hotspot:

- the host laptop often appears as `192.168.137.1`
- edge devices should use that IP in the `Central Server URL`

Telegram alerting

The project can send Telegram notifications for:

- activity changes
- falls
- major fall burst alerts

To enable Telegram:

1. Create a bot with BotFather
2. Get the bot token
3. Get the destination chat ID
4. Open the settings page
5. Enter:
   - bot token
   - chat ID
   - enable Telegram
6. Use the test alert button on the settings page

Data storage

The system stores:

- activity summaries by date and person
- fall history with timestamps
- face encodings
- ReID identity bank
- manual naming mappings
- system settings

This allows the project to preserve important state across restarts.

Evaluation and validation

The project includes live operational metrics such as:

- FPS
- people tracked
- falls today
- major falls today
- low-power entries
- wake events
- wake latency
- connected nodes
- sync success and failure counts

However, formal activity and fall accuracy still need to be measured on labeled test videos.

Recommended validation method:

1. Record several videos with different activities and environments
2. Label each activity segment by time
3. Run the system on those videos or replay scenarios live
4. Compare the predictions against the ground truth
5. Compute:
   - accuracy
   - precision
   - recall
   - F1-score
   - false alarm rate
   - detection delay

This is the correct way to report project-level performance in a presentation or report.

Current strengths of the project

- real-time live monitoring
- strong demo value
- integrated dashboard
- practical alerting
- identity persistence
- low-power optimization
- edge/server architecture
- caregiver-oriented summary features
- GPU-capable runtime

Current limitations

- most logic is in a single large Python file
- project-level accuracy benchmarking is not yet formalized
- fall and activity logic are heuristic-based rather than learned end-to-end
- distributed deployment works, but can still be refined for cleaner production structure

Suggested future improvements

- split the application into separate `server.py` and `edge.py`
- add formal offline evaluation tooling for labeled videos
- add exportable CSV or PDF reports
- add room transition history across nodes
- add trend analysis over multiple days
- add configurable risk scoring
- add more explicit health and wellness anomaly detection

How to present the project honestly

A good presentation-safe explanation is:

“This system uses a pretrained pose estimation model as its AI backbone, then applies custom temporal and posture-based logic to detect activity states and fall events. It also includes alerting, low-power optimization, a web dashboard, and multi-node monitoring support.”

Troubleshooting

If the dashboard does not open:

- check that the Flask server started successfully
- confirm the configured port
- confirm local firewall rules if accessing from another device

If the edge laptop cannot connect:

- confirm both devices are on the same network
- confirm the correct server IP address
- confirm `0.0.0.0` bind host on the server laptop
- confirm firewall allows the chosen port

If the app runs slowly:

- make sure CUDA-enabled PyTorch is installed
- confirm `torch.cuda.is_available()` returns `True`
- use a smaller preview window
- reduce dashboard stream size or FPS if needed

If pose weights are missing:

- place `yolo11n-pose.pt` in the project folder
- or allow the first-time model download

Author note

This project is best understood as an applied intelligent monitoring system rather than only a machine learning model. Its value comes from combining pose AI, decision logic, persistence, alerting, optimization, and practical usability into a single working care-oriented platform.
#   6 t h - f a l l  
 