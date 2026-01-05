# AI Photo Site - Biometric Photo Quality Control System

A modern, AI-powered biometric photo quality control and analysis platform. Automatically evaluates photo quality using FastAPI, OpenCV, and MediaPipe.

## Features

- Photo Upload: Support for JPG, PNG, and WEBP formats
- AI Analysis: Automatic image analysis using OpenCV and MediaPipe
- Quality Control: Face detection, blur detection, brightness analysis, and framing validation
- Modern UI: PhotoAid-style modal flow with real-time progress indicators
- Real-time Processing: Asynchronous analysis using background tasks
- Detailed Reporting: PASS/FAIL results with specific reasons

## Quick Start

### Requirements

- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai-photo-site
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate the virtual environment:**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues installing MediaPipe, ensure all required system dependencies are installed.

### Running the Application

```bash
uvicorn app:app --reload
```

The application will be available at `http://localhost:8000`.

## Project Structure

```
ai-photo-site/
├── app.py                 # FastAPI main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── templates/            # Jinja2 HTML templates
│   ├── index.html        # Main page
│   ├── job.html         # Job status page
│   └── ...
├── static/               # Static files (CSS, JS)
│   ├── styles.css       # Custom styles
│   └── app.js           # Frontend JavaScript
└── uploads/              # Uploaded photos (gitignore)
```

## Usage

1. Open `http://localhost:8000` in your browser
2. Click "Before You Start" to read the guidelines
3. Select a photo and click "Upload Photo"
4. Watch the AI analysis progress in real-time
5. View PASS/FAIL status and detailed results

## Analysis Criteria

### PASS Criteria
- Single face detected
- Face is sharp and in focus
- Adequate lighting
- Proper framing

### FAIL Criteria (Cannot be fixed by AI)
- No face detected
- Multiple faces in photo
- Photo is too blurry
- Face is too dark or overexposed
- Face framing is inappropriate

### Automatically Fixed (Not shown to user)
- Background replacement (white background)
- Aspect ratio correction (50x60mm)
- Exposure balance
- Minor tilt corrections

## Technology Stack

**Backend:**
- FastAPI - Modern Python web framework
- OpenCV - Image processing
- MediaPipe - Face detection
- NumPy - Numerical computations

**Frontend:**
- HTML5 / CSS3
- JavaScript (Vanilla)
- Tailwind CSS - Utility-first CSS framework
- Jinja2 - Template engine

## API Endpoints

- `GET /` - Main page
- `POST /upload` - Photo upload
- `GET /job/{job_id}` - Job status page
- `GET /job/{job_id}/status` - Job status (JSON)
- `GET /uploads` - Uploaded files list

## Configuration

Analysis threshold values can be adjusted in `app.py`:

```python
FACE_BLUR_THRESHOLD = 50.0
FACE_BRIGHTNESS_MIN = 50.0
FACE_BRIGHTNESS_MAX = 240.0
FACE_RATIO_MIN_UNRECOVERABLE = 0.05
FACE_RATIO_MAX_UNRECOVERABLE = 0.60
MIN_RESOLUTION = 400 * 400
```

## License

This is a private project.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions, please open an issue.

---

**Note:** This project is under active development. Additional testing and optimization may be required for production use.
