# Algorithmic AI Director (Reachy Mini)

A Reachy Mini that watches you film a 15–60 s Reel/TikTok, sends the whole take to a
Vision-Language Model, and then roasts your retention metrics out loud while nodding, shaking
its head, or turning away in disgust as if swiping to the next video.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env            # then add GEMINI_API_KEY or ANTHROPIC_API_KEY
python main.py
```

Say **"Reachy, rolling!"**, perform, say **"Cut!"**. In the preview window you can also press
**R** (rolling), **C** (cut), **Q** (quit). Ctrl+C shuts everything down cleanly.

Use `VLM_PROVIDER=mock` to test the robot, voice and state machine with no API key.

**Physical vs simulation.** On start-up the HAL tries `ReachyMini(use_sim=False)`. If no robot
or daemon answers, it calls `ReachyMini(spawn_daemon=True, use_sim=True)`, which starts the MuJoCo
simulator. (In the SDK, `use_sim` only takes effect together with `spawn_daemon=True`.) If both
fail, the app keeps running "headless" with your webcam, so you can still work on the
prompt and voice.

## Project layout

```
main.py                     entry point (logging + DirectorApp)
ai_director/
  config.py                 every tunable, from env / .env
  state.py                  thread-safe FSM: IDLE → RECORDING → PROCESSING → FEEDBACK → IDLE
  messages.py               queue message types + CritiqueResult (pydantic JSON schema)
  hal.py                    RobotHAL: physical → sim → headless; camera; emotion choreographies
  prompts.py                the critic's system prompt + per-take context
  tts_worker.py             pyttsx3 in a subprocess (thread-safe, killable)
  app.py                    supervisor: wiring, preview window, watchdog, clean shutdown
  vlm/clients.py            Gemini / Anthropic / Mock clients, all forced to strict JSON
  nodes/
    base.py                 BaseNode thread (crash isolation, auto-restart, shutdown hooks)
    vision_node.py          camera loop, take buffer, face tracking
    audio_node.py           mic capture + recognizer worker, wake words, transcript
    brain_node.py           VLM call with hard timeout, retries, "brain offline" fallback
    action_node.py          TTS ∥ motion, returns FSM to IDLE
```

## Data flow

```
             ┌──────────────── StateMachine (shared, lock + condition) ────────────────┐
             │   IDLE ──rolling──▶ RECORDING ──cut──▶ PROCESSING ──▶ FEEDBACK ──▶ IDLE   │
             └──────────────────────────────────────────────────────────────────────────┘
 mic ─▶ AudioNode ─(TakeEnded: take_id, duration, timestamped transcript)─▶ brain_queue ─▶ BrainNode
          │                                                                     │  ▲
          │ PoseCommand('attention')                  vision.export_take(id) ───┘  │
          ▼                                                                        │
     action_queue ◀─(PoseCommand('thinking'), CritiqueCommand(result))─────────────┘
          │
          ▼
     ActionNode ──▶ TTS subprocess  ∥  RobotHAL.play_emotion()  ──▶ FSM: FEEDBACK → IDLE

 camera ─▶ VisionNode (20 Hz live loop; 2 fps JPEG take buffer while RECORDING; face tracking)
```

1. **Audio → FSM.** The mic thread hands raw phrases to a recognizer thread through a private
   queue, so the mic keeps listening while a phrase is being transcribed. In IDLE, "rolling"
   moves the FSM to RECORDING and queues an `attention` gesture.
2. **Vision buffers the take.** The vision loop sees RECORDING, clears the buffer, starts face
   tracking (native daemon tracking if available, otherwise OpenCV and `look_at_pixel`), and
   keeps 2 timestamped JPEGs per second. Very long takes are thinned out evenly rather than cut
   short, so the hook is never lost.
3. **"Cut!"** The Audio node switches the FSM to PROCESSING and puts a `TakeEnded` (with the
   timestamped transcript) on `brain_queue`.
4. **Brain.** It queues a looping `thinking` sway, pulls the frames with
   `vision.export_take(take_id)` (thinned evenly to ≤ `MAX_FRAMES_TO_VLM`, always keeping the
   first and last frame), and calls the VLM in a worker thread with a hard timeout and retries.
   The reply is validated against `CritiqueResult`. Any failure (timeout, auth, network, bad
   JSON) becomes a canned "my brain is offline" critique. It then sets FEEDBACK and queues a
   `CritiqueCommand`.
5. **Action.** It speaks `spoken_critique` (TTS subprocess on a helper thread) while performing
   `reachy_emotion` on its own thread. A FAIL always gets at least the "swipe away" move. It then
   returns the FSM to IDLE. Phrases the mic caught while the robot was talking are discarded.

Robot motion never blocks the camera: motion and camera use separate locks, VLM calls run in
their own executor, and face-tracking ticks are skipped while an expressive move owns the head.
A watchdog in the supervisor resets to IDLE if any state gets stuck.

## Troubleshooting

- **Red dot / "error occurred" on the dashboard (http://127.0.0.1:8000):** the simulator's
  MuJoCo backend missed the daemon's 2-second start-up window. The app now launches the
  simulator itself, detects this and restarts it (up to 3 tries). Simulator logs go to
  `logs/daemon.log`. If it keeps failing, set `SIM_HEADLESS=true`.
- **`cv2 has no attribute CascadeClassifier`:** you have OpenCV 5. Run
  `pip install "opencv-python>=4.9,<5"`.
- **R / C / Q keys:** press them in the terminal running `python main.py`, or in the
  "AI Director" preview window. In PyCharm's Run tool window, enable *Emulate terminal in
  output console* or use the Terminal tab.
- **Python 3.10** is limited to `reachy-mini` 1.8.x. Python 3.11 or newer gets the latest SDK.

## Tuning notes

- **Emotion moves** are plain keyframe tables in `hal.py` (`CHOREOGRAPHIES`). If "look down"
  looks up on your unit, flip `PITCH_SIGN`. Set `USE_RECORDED_EMOTIONS=true` to use Pollen's
  recorded emotion clips instead, chosen by keyword.
- **Microphone:** set `MIC_NAME_HINT=Reachy` (or `MIC_DEVICE_INDEX`) to use the robot's mic
  array. Speech recognition uses Google's free web recognizer, so it needs internet access.
- **Debugging the critic:** set `SAVE_TAKES_DIR=takes` to keep exactly the frames and
  transcript that were sent to the model.
