 # 🚨 SirenSight – AI-Based Eye Closure Detection & Alert System
  
  ## 📌 Overview
  
  SirenSight is a real-time drowsiness and focus monitoring system that leverages computer vision to detect prolonged eye closure and trigger escalating alerts. It is designed to promote safety and sustained attention in high-focus environments
  such as driving, studying, or working late hours.
  
  ## 🔑 Key Features
  
  * **👁️ Real-Time Eye Closure Detection** – Continuously monitors the user's eyes through a webcam and detects closure using computer vision techniques.
  * **⏱️ Duration-Based Monitoring** – Accurately measures how long the eyes remain closed, tracking every second to detect signs of fatigue or inattention.

  * **🚨 Tiered Alert System**
  * – Issues progressively louder alerts based on how long the eyes stay closed:
    * 🟡 Mild siren after 3 seconds
    * 🔴 Intense alert sound after 4 seconds
  * - Helps users stay alert, reducing the risk of microsleep or distractions, especially in critical or productivity-driven situations.
  
  ## 🛠️ Technologies Used
  * 🐍 Python – Core programming language
  * 📸 OpenCV – For real-time video capture and frame processing
  * 🧩 Dlib or Haar Cascades – Eye and facial feature detection
  * 🔊 Pygame / playsound – To play warning sounds
  * 🧵 Threading & Time modules – For non-blocking timing and alert logic
  
  ## 📌 Use Case Scenarios
  * 🚗 Driver Drowsiness Detection
  * 🎓 Student Focus Monitoring
  * 🧑‍💻 Work-from-Home Fatigue Prevention
  
  ## 🚀 How to Run
  
  1. Clone the repository
     ```bash
     git clone https://github.com/34anjani/SirenSight.git
     cd SirenSight
     ```
  
  2. Install dependencies
      ```bash
     pip install opencv-python dlib pygame numpy
      ```
  
  3. Download the shape predictor model
     Download shape_predictor_68_face_landmarks.dat from:
     http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
     > Extract and place it in the project root folder
  
  4. Run the application
     ```bash
     python modify.py
     ```
  
  5. Usage
     
  - Ensure your webcam is connected
  - The system will start monitoring your eyes in real-time
  - Close your eyes for 3+ seconds to trigger the alert
  
