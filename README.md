# Thaal Counter - Computer Vision Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A computer vision application designed specifically for mosque kitchens to automatically track and count metal serving plates (thaals) as they move between the kitchen and dining hall.

## What is Thaal Counter?

The Thaal Counter is an intelligent tracking system that uses computer vision and machine learning to monitor thaal flow in mosque kitchens. It provides real-time counting, data analytics, and automated inventory management for community meal services.

### Key Features

- **Real-time Object Detection**: Custom-trained YOLO11 model for accurate thaal recognition
- **Line Crossing Detection**: Automatic counting of thaals going out (kitchen to hall) and coming in (hall to kitchen)
- **Live Video Monitoring**: Real-time video feed with visual tracking and counting
- **Secure Authentication**: 4-digit passcode protection with session management
- **Data Analytics Dashboard**: Historical analysis with charts, metrics, and export capabilities
- **Mobile-Responsive Interface**: Optimized for both desktop and mobile devices
- **Session Management**: Start/stop counting with registered thaal count tracking
- **Export Capabilities**: Download data for record keeping and analysis

### Technology Stack

- **Backend**: Python, Flask, SQLite
- **Computer Vision**: OpenCV, Ultralytics YOLO11
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Hardware**: Raspberry Pi 5, USB Webcam
- **Database**: SQLite with custom schema for events and sessions

## Why Thaal Counter?

Managing thaal inventory during community meals can be challenging. The kitchen volunteers had a hard time keeping track of how many thaals were used during large events in a fast pace environment. The main problem was disagreements in the number of thaals served with the cleaners who charged on a per thaal basis. This system eliminates manual counting, reduces human error, and provides real-time data to help kitchen staff efficiently manage the flow of serving plates during busy meal times. It's also the beginning of data analysis for the kitchen!

### Problem Solved

- **Manual Counting Errors**: Eliminates human error in thaal counting
- **Time Management**: Reduces staff time spent on inventory tracking
- **Data Insights**: Provides analytics for better kitchen operations
- **Resource Optimization**: Helps optimize thaal usage and cleaning cycles
- **Accountability**: Maintains accurate records of thaal flow

### Impact

- Improved efficiency in mosque kitchen operations
- Better resource management and planning
- Data-driven insights for community meal services
- Reduced manual labor and human error
- Enhanced transparency in kitchen operations

## Where is it Deployed?

### Hardware Environment

- **Primary Platform**: Raspberry Pi 5 with USB webcam
- **Camera Position**: Mounted to monitor the counter area between kitchen and dining hall
- **Network**: Local network access with SSH capability
- **Storage**: Local SQLite database for data persistence

### Physical Setup

The system is strategically positioned to monitor the counter area where thaals are placed when moving between:
- **Kitchen Preparation Area**: Where thaals are loaded with food
- **Dining Hall**: Where community members eat from thaals
- **Return Area**: Where used thaals are placed for cleaning

### System Architecture

```
Raspberry Pi 5
├── Camera Input (USB Webcam)
├── Computer Vision Pipeline
│   ├── YOLO11 Model Inference
│   ├── Object Tracking
│   └── Line Crossing Detection
├── Web Application (Flask)
│   ├── Authentication System
│   ├── Real-time Monitoring
│   └── Data Dashboard
└── Database (SQLite)
    ├── Events Table
    └── Sessions Table
```

## When is it Active?

### Operational Timing

- **Service Periods**: Activated during meal service times (lunch and dinner)
- **Flexible Scheduling**: Staff can start/stop counting as needed
- **Session Management**: Each service session is tracked independently
- **Data Retention**: Historical data is maintained for analysis

### Service Lifecycle

1. **Session Start**: Staff enters expected thaal count and starts service
2. **Active Monitoring**: System tracks all thaal movements in real-time
3. **Data Collection**: Events are logged with timestamps and session IDs
4. **Session End**: Staff stops service, final counts are recorded
5. **Data Analysis**: Historical data is available for dashboard analysis

### Data Management

- **Real-time Processing**: Events are processed and stored immediately
- **Historical Analysis**: Data is available for trend analysis and reporting
- **Export Capabilities**: Data can be exported for external analysis
- **Session Tracking**: Each service session maintains complete audit trail

## How Does it Work?

### Computer Vision Pipeline

1. **Image Capture**: USB webcam captures video frames at 1280x720 resolution
2. **Object Detection**: Custom-trained YOLO11 model identifies thaals in each frame
3. **Object Tracking**: Ultralytics tracking assigns unique IDs to detected thaals
4. **Line Crossing Detection**: Algorithm detects when thaals cross predefined lines
5. **Event Logging**: Crossing events are timestamped and stored in database

### YOLO Model Training

- **Dataset**: 182 labeled images (129 training, 38 validation, 15 test)
- **Model**: YOLO11n architecture fine-tuned for thaal detection
- **Training**: Custom dataset with Roboflow integration
- **Performance**: Optimized for real-time inference on Raspberry Pi 5

### Line Crossing Algorithm

```python
# Simplified line crossing logic
if prev_x > LINE_OUT_POSITION and center_x <= LINE_OUT_POSITION:
    thaal_out_count += 1
elif prev_x < LINE_IN_POSITION and center_x >= LINE_IN_POSITION:
    thaal_in_count += 1
```

### Database Schema

**Events Table:**
- `id`: Primary key
- `timestamp`: Event timestamp
- `event_type`: THAAL_OUT, THAAL_IN, SERVICE_START, SERVICE_STOP
- `session_id`: Associated session identifier

**Sessions Table:**
- `session_id`: Primary key
- `start_time`: Session start timestamp
- `end_time`: Session end timestamp
- `expected_thaals`: Expected thaal count
- `final_thaals_out`: Final outbound count
- `final_thaals_in`: Final inbound count

## Installation & Setup

### Prerequisites

- Raspberry Pi 5 (or compatible single-board computer)
- USB Webcam (compatible with OpenCV)
- Python 3.8 or higher
- 8GB+ microSD card
- Stable internet connection

### Hardware Requirements

- **Processor**: ARM64 architecture (Raspberry Pi 5 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 16GB minimum for OS and application
- **Camera**: USB webcam with 720p+ resolution
- **Network**: WiFi or Ethernet connection

### Software Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/kitchen-cv.git
cd kitchen-cv
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install system dependencies:**
```bash
sudo apt update
sudo apt install python3-opencv libopencv-dev
```

### Environment Configuration

1. **Set up environment variables:**
```bash
export THAAL_COUNTER_PASSCODE="your-4-digit-passcode"
export THAAL_COUNTER_SECRET_KEY="your-secure-secret-key"
```

2. **Generate a secure secret key:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

3. **Make environment variables permanent:**
```bash
echo 'export THAAL_COUNTER_PASSCODE="your-passcode"' >> ~/.bashrc
echo 'export THAAL_COUNTER_SECRET_KEY="your-secret-key"' >> ~/.bashrc
source ~/.bashrc
```

### Database Initialization

The database will be automatically created on first run. To manually initialize:

```bash
python3 -c "import database; database.init_db()"
```

## Usage Guide

### Starting the Application

1. **Launch the application:**
```bash
python3 app.py
```

2. **Access the web interface:**
   - Open browser to `http://your-pi-ip:5001`
   - Enter 4-digit passcode to access the system

### Service Management

1. **Start a Service Session:**
   - Click "Start Service" button
   - Enter expected thaal count
   - System begins real-time monitoring

2. **Monitor Live Feed:**
   - View real-time video with tracking visualization
   - Watch thaal counts update in real-time
   - See line crossing detection in action

3. **Stop Service Session:**
   - Click "Stop Service" button
   - Final counts are recorded
   - Session data is saved for analysis

### Dashboard Analytics

1. **Access Dashboard:**
   - Click "View Dashboard" from main interface
   - Select date to view historical data

2. **View Analytics:**
   - Cumulative flow charts
   - Throughput per minute graphs
   - Service duration metrics
   - Discrepancy analysis

3. **Export Data:**
   - Export daily event logs as CSV
   - Download session summaries
   - Generate reports for analysis

## Technical Details

### Project Structure

```
kitchen-cv/
├── app.py                 # Main Flask application
├── vision_processor.py    # Computer vision pipeline
├── database.py           # Database operations
├── templates/            # HTML templates
│   ├── login.html        # Authentication page
│   ├── main.html         # Main control interface
│   └── dashboard.html    # Analytics dashboard
├── labeled_dataset/      # Training data
│   ├── train/           # Training images
│   ├── valid/           # Validation images
│   └── test/            # Test images
├── runs/detect/train/   # Trained model weights
└── events.db            # SQLite database
```

### Key Files

- **`app.py`**: Flask web application with authentication and API endpoints
- **`vision_processor.py`**: Computer vision processing and object tracking
- **`database.py`**: Database operations and data management
- **`templates/`**: Frontend HTML templates with responsive design
- **`labeled_dataset/`**: YOLO training dataset with annotations

### API Endpoints

- `GET /` - Main control interface
- `GET /dashboard` - Analytics dashboard
- `POST /login` - Authentication
- `POST /start_service` - Start counting session
- `POST /stop_service` - Stop counting session
- `GET /status` - Current system status
- `GET /api/dashboard_data` - Historical data
- `GET /export_by_date` - Export daily data
- `GET /export_sessions` - Export session data

### Model Training Information

- **Framework**: Ultralytics YOLO11
- **Architecture**: YOLO11n (nano) for Raspberry Pi optimization
- **Dataset Size**: 182 images total
- **Training Split**: 70% train, 21% validation, 9% test
- **Classes**: Single class ('thaal')
- **Annotation Format**: YOLO format with bounding boxes
- **Training Platform**: Roboflow integration

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `THAAL_COUNTER_PASSCODE` | 4-digit access code | `1234` |
| `THAAL_COUNTER_SECRET_KEY` | Flask session secret | `your-secret-key` |

### Camera Configuration

The system automatically detects and configures USB webcams. For custom camera settings, modify `vision_processor.py`:

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Line Position Adjustment

Adjust detection lines in `vision_processor.py`:

```python
self.LINE_OUT_POSITION = self.FRAME_WIDTH // 3
self.LINE_IN_POSITION = (self.FRAME_WIDTH // 3) * 2
```

## License & Credits

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Developer

**Aamir Merchant**
- LinkedIn: [merchant-aamir](https://www.linkedin.com/in/merchant-aamir/)
- Email: [aamirkmerchant@gmail.com](mailto:aamirkmerchant@gmail.com)

### Acknowledgments

- **Ultralytics YOLO**: For the computer vision framework
- **Flask**: For the web application framework
- **OpenCV**: For computer vision processing
- **Chart.js**: For dashboard visualizations
- **Roboflow**: For dataset management and annotation tools

### Dataset Credits

The training dataset was created using Roboflow and is available at:
[Thaal Counter Dataset](https://universe.roboflow.com/thaal-counter/thaal-counter-k2ty8/dataset/2)

---

**Support**: For troubleshooting or technical support, please reach out to [aamirkmerchant@gmail.com](mailto:aamirkmerchant@gmail.com)

*This project demonstrates the power of AI and computer vision in solving real-world community challenges, making daily operations more efficient and data-driven.*