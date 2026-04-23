Act as a professional technical presenter. Generate a 10-slide professional presentation for the project "Real-Time Restricted Area Monitoring System with YOLO". The presentation should cover technical aspects, architecture, use cases, and how to execute the project. Use clean, professional language, clear headings, and bullet points. Assume the audience consists of technical reviewers and stakeholders.

Here are the specific requirements for each slide:

**Slide 1: Title & Introduction**
- Title: Real-Time Restricted Area Monitoring System with YOLO
- Subtitle: Continuous streaming, detection, and alerts using Computer Vision
- Include a brief introduction: This project uses YOLOv8 for real-time object detection, identifying unauthorized intrusions in a restricted area, and logging the violations in real-time on a web dashboard.

**Slide 2: Problem Statement & Objectives**
- Highlight the problem: Physical security and safety monitoring often rely on manual supervision, which is prone to human error and fatigue.
- State the objective: Develop an automated, real-time intrusion and object monitoring system that can instantly detect when objects enter a user-defined restricted boundary and trigger auditory and visual alerts.

**Slide 3: System Architecture & Workflow**
- Explain the flow of the system:
  1. The webcam captures live video frames.
  2. The YOLOv8 model processes each frame to detect defined objects.
  3. The system maps the objects against a virtual restricted boundary (an ellipse drawn on the frame).
  4. If an object breaches the boundary, an alert sounds, and the event logs to a CSV database.
  5. A real-time WebSocket connection pushes data instantly to a web analytics dashboard.

**Slide 4: Core Technologies Used**
- Programming Language: Python
- Computer Vision & AI: Ultralytics YOLOv8 for high-speed object detection and OpenCV for video processing and drawing bounding boxes/boundaries.
- Frontend Streaming: Streamlit for an interactive user interface, allowing dynamic model configuration and real-time video playback.
- Backend & Dashboard: FastAPI to serve an HTML/JS analytics dashboard and manage WebSockets for instant data synchronization. Data handled via Pandas.

**Slide 5: Key Features & Capabilities**
- Define specific features of the application:
  - Real-time video streaming with overlaid bounding boxes and confidence scores.
  - Customizable objects of interest and alert-triggering targets.
  - Euclidean distance-based logic to verify if objects have crossed the restricted threshold.
  - Automatic auditory alerts (via Pygame) for immediate, on-site intrusion warnings.

**Slide 6: Dynamic UI & Configuration (Streamlit View)**
- Describe the configuration controls available to the user:
  - Select between different YOLO models (e.g., standard YOLOv8n vs Intrustion specific weights).
  - Adjust confidence threshold dynamically to filter out false positives.
  - Multi-select capability to detect multiple distinct objects simultaneously.
  - One-click buttons to start or stop the webcam securely.

**Slide 7: Data Logging & Analytics (FastAPI Dashboard)**
- Elaborate on the data storage and web analytics platform:
  - Violations are written to `detection_log.csv` capturing timestamp, object class, confidence, and violation status.
  - FastAPI powers a dashboard showing total detections, total violations, and the most frequent violating class.
  - The dashboard automatically pulls updates every second utilizing asynchronous WebSocket architecture, requiring zero manual page refresh.

**Slide 8: How to Execute & Run the System**
- Break down the 3-step execution process:
  1. Installation: Install necessary python packages via `pip install -r requirements.txt`.
  2. Start the Backend Dashboard: Run `uvicorn fastapi_run:app --reload` to start the FastAPI server on port 8000 handling WebSockets.
  3. Launch the Application: Run `streamlit run streamlit_run.py` to open the streaming interface and start the camera.

**Slide 9: Real-World Applications**
- List 4 practical use cases:
  - Industrial Safety: Ensuring workers don't enter hazardous machine operating zones.
  - Security Surveillance: Monitoring secure facilities and restricted perimeters after hours.
  - Retail/Private Properties: Preventing unauthorized access to staff-only areas.
  - PPE Compliance: Adapting the system to ensure correct gear is worn before entry.

**Slide 10: Conclusion & Future Scope**
- Summarize the project's impact: A scalable, low-latency, hybrid web application seamlessly combining AI vision models with live data telemetry.
- Outline future enhancements: 
  - Integration with cloud databases (like PostgreSQL/MongoDB) instead of local CSV.
  - Notifications via SMS or Email (Twilio/SendGrid integration).
  - Multi-camera IP stream support for wider geographic monitoring.
