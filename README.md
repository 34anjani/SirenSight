🚨 SirenSight – AI-Based Eye Closure Detection & Alert System

SirenSight is a real-time drowsiness and focus monitoring system that leverages computer vision to detect prolonged eye closure and trigger escalating alerts. It
is designed to promote safety and sustained attention in high-focus environments such as driving, studying, or working late hours.

🔑 Key Features

      👁️ Real-Time Eye Closure Detection
      
         Continuously monitors the user’s eyes through a webcam and detects closure using computer vision techniques.
        
      ⏱️ Duration-Based Monitoring
      
         Accurately measures how long the eyes remain closed, tracking every second to detect signs of fatigue or inattention.
        
      🚨 Tiered Alert System
      
         Issues progressively louder alerts based on how long the eyes stay closed:
        
            🟡 Mild siren after 3 seconds
            
            🔴 Intense alert sound after 4 seconds
            
         Helps users stay alert, reducing the risk of microsleep or distractions, especially in critical or productivity-driven situations.

🛠️ Technologies Used

    🐍 Python – Core programming language
    
    📸 OpenCV – For real-time video capture and frame processing
    
    🧩 Dlib or Haar Cascades – Eye and facial feature detection
    
    🔊 Pygame / playsound – To play warning sounds
    
    🧵 Threading & Time modules – For non-blocking timing and alert logic
    
📌 Use Case Scenarios

    🚗 Driver Drowsiness Detection
    
    🎓 Student Focus Monitoring
    
    🧑‍💻 Work-from-Home Fatigue Prevention

  # 🚀 How to Run
  
  1. Clone the repository
     ```bash
     git clone https://github.com/34anjani/SirenSight.git
     cd SirenSight
  
  2. Install dependencies
  
     pip install opencv-python dlib pygame numpy
  
  3. Download the shape predictor model
  
     Download shape_predictor_68_face_landmarks.dat from:
     http://dlib.net/files/shape_predictor_68_face_landmarks_GTK_FILE_DIALOG.dat.bz2
  
     > Extract and place it in the project root folder
  
  4. Run the application
  
     python modify.py
  
  5. Usage
    - Ensure your webcam is connected
    - The system will start monitoring your eyes in real-time
    - Close your eyes for 3+ seconds to trigger the alert


 # 🚨 SirenSight – AI-Based Eye Closure Detection & Alert System
  
  ## 📌 Overview
  
  Drowsy driving and fatigue-related accidents are a leading cause of road fatalities. SirenSight is a **real-time drowsiness monitoring system** that uses **computer vision** to detect prolonged eye closure and triggers escalating audio alerts
  to keep users alert.
  
  The system continuously tracks facial landmarks through a webcam, calculates the Eye Aspect Ratio (EAR), and issues progressively louder sirens based on how long the eyes remain closed — making it suitable for driving, studying, or working
  late hours.
  
  ## 🚀 Key Highlights
  * Real-time eye closure detection using webcam
  * Duration-based fatigue monitoring (tracks every second)
  * Tiered alert system with escalating intensity
  * Lightweight and runs on any standard laptop
  * Applicable to driving safety, study focus, and WFH fatigue prevention
  
  ## 🧠 How It Works
  
  The detection pipeline consists of the following steps:
  * **Face Detection** – Detects face in each video frame using Haar Cascades or Dlib
  * **Landmark Detection** – Extracts 68 facial landmarks to isolate eye regions
  * **EAR Calculation** – Computes Eye Aspect Ratio to determine if eyes are closed
  
  ### 🔊 Alert Levels
  * 🟡 **Mild siren** after 3 seconds of eye closure
  * 🔴 **Intense alert** after 4 seconds of continuous closure
  
  ## 🗂️ Input / Output
  * **Input**: Live webcam video feed
  * **Output**: Real-time eye status monitoring with audio alerts on fatigue detection
  
  ## 🛠️ Tech Stack
  * Python
  * OpenCV – Real-time video capture and frame processing
  * Dlib / Haar Cascades – Eye and facial feature detection
  * Pygame – Audio alert playback
  * Threading & Time modules – Non-blocking timing and alert logic
  
  ## 📌 Use Case Scenarios
  * 🚗 Driver Drowsiness Detection
  * 🎓 Student Focus Monitoring
  * 🧑‍💻 Work-from-Home Fatigue Prevention
  
  ## 🚀 How to Run
  
  1. Clone the repository
     ```bash
     git clone https://github.com/34anjani/SirenSight.git
     cd SirenSight
  
  2. Install dependencies
  
     pip install opencv-python dlib pygame numpy
  
  3. Download the shape predictor model
  
     Download shape_predictor_68_face_landmarks.dat from:
     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  
     > Extract and place it in the project root folder
  
  4. Run the application
  
     python modify.py
  
  5. Usage
    - Ensure your webcam is connected
    - The system will start monitoring your eyes in real-time
    - Close your eyes for 3+ seconds to trigger the alert

