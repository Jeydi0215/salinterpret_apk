"""
ASL Translator - Optimized Version
Focus on performance improvements and faster loading
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to speed up loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty, ListProperty, BooleanProperty
from kivy.config import Config
import cv2
import numpy as np
import time
import threading
import math
import queue
from functools import partial

# Configure Kivy for better performance
Config.set('graphics', 'maxfps', '30')  # Limit max FPS
Config.set('graphics', 'multisamples', '0')  # Disable multisampling
Config.set('kivy', 'desktop', '1')
Config.set('kivy', 'exit_on_escape', '1')

# Create a simple logger function that doesn't block
log_queue = queue.Queue()
def log(message):
    """Non-blocking logger that puts messages in a queue"""
    timestamp = time.strftime("%H:%M:%S")
    log_queue.put(f"[{timestamp}] {message}")
    
    # Periodically write logs to disk in a separate thread
    if not hasattr(log, "initialized"):
        log.initialized = True
        threading.Thread(target=log_writer, daemon=True).start()

def log_writer():
    """Writes logs from queue to file in background"""
    with open('asl_log.txt', 'w') as f:
        while True:
            try:
                messages = []
                # Get as many messages as possible
                while not log_queue.empty() and len(messages) < 100:
                    messages.append(log_queue.get_nowait())
                
                if messages:
                    for msg in messages:
                        f.write(msg + "\n")
                        print(msg)
                    f.flush()
                    
                time.sleep(0.5)  # Sleep to reduce CPU usage
            except Exception as e:
                print(f"Log error: {e}")
                time.sleep(1)

class ASLTranslatorWidget(BoxLayout):
    """Root widget for the ASL Translator application"""
    
    # Properties that can be bound to the UI
    current_letter = StringProperty("Waiting...")
    confidence = NumericProperty(0.0)  # Make sure this is initialized as 0.0
    translated_word = StringProperty("")
    status_color = ListProperty([1, 0, 0, 1])  # RGBA, red by default
    hand_detected = BooleanProperty(False)
    loading_progress = NumericProperty(0)  # 0-100 progress indicator
    
    def __init__(self, **kwargs):
        super(ASLTranslatorWidget, self).__init__(**kwargs)
        log("Initializing ASL Translator")
        
        # Initialize variables
        self.is_running = True
        self.current_word = []
        self.model = None
        self.labels = None
        self.cap = None
        self.last_prediction_time = 0
        self.prediction_interval = 1  # seconds
        self.imgSize = 224
        self.offset = 40
        self.model_loaded = threading.Event()  # Signal when model is loaded
        
        # IMPROVEMENT: Use a smaller frame buffer to decrease latency
        self.frame_queue = queue.Queue(maxsize=2)  # For passing frames to processing
        self.result_queue = queue.Queue()  # For passing prediction results
        
        # Create camera image
        self.camera_image = Image(allow_stretch=True, keep_ratio=True)
        self.ids.camera_container.add_widget(self.camera_image)
        
        # Show loading screen immediately
        self.show_loading_screen()
        
        # IMPROVEMENT: Start camera initialization immediately
        # while loading model in parallel
        threading.Thread(target=self.init_camera, daemon=True).start()
        
        # IMPROVEMENT: Load model in separate thread to avoid blocking UI
        threading.Thread(target=self.load_model_thread, daemon=True).start()
        
        # Start the display updater with a slower refresh rate
        Clock.schedule_interval(self.update_display, 1/30)  # 30 FPS is plenty
        
        # IMPROVEMENT: Update loading progress animation
        Clock.schedule_interval(self.update_loading_animation, 0.2)
    
    def update_loading_animation(self, dt):
        """Updates the loading animation while components initialize"""
        if not self.model_loaded.is_set():
            # Rotate through 0-100 for progress indicator
            self.loading_progress = (self.loading_progress + 5) % 100
    
    def show_loading_screen(self):
        """Show a loading screen while initializing"""
        # Create a loading image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Loading ASL Translator...", (120, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Please wait, initializing components", (80, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(img, "This may take a few moments", (150, 280), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Convert to texture directly
        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_image.texture = texture
    
    def load_model_thread(self):
        """Load the model in a background thread"""
        # Update UI to show we're loading the model
        Clock.schedule_once(lambda dt: setattr(self, 'current_letter', "Loading model..."), 0)
        
        # Load the model in background
        log("Loading TensorFlow model...")
        success = self.load_model()
        
        if not success:
            Clock.schedule_once(lambda dt: setattr(self, 'current_letter', "Error: Model loading failed"), 0)
            log("Model loading failed")
            return
        
        # Now load the detector
        Clock.schedule_once(lambda dt: setattr(self, 'current_letter', "Initializing detector..."), 0)
        if not self.init_detector():
            Clock.schedule_once(lambda dt: setattr(self, 'current_letter', "Error: Detector initialization failed"), 0)
            log("Detector initialization failed")
            return
        
        # Signal that critical components are loaded
        self.model_loaded.set()
        
        # Start processing thread
        threading.Thread(target=self.processing_loop, daemon=True).start()
        
        # Update UI when everything is ready
        Clock.schedule_once(lambda dt: setattr(self, 'current_letter', "Ready"), 0)
        log("Model and detector initialization complete")
    
    def load_model(self):
        """Load the TensorFlow model and labels"""
        try:
            # IMPROVEMENT: Lazy import TensorFlow only when needed
            log("Importing TensorFlow...")
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            # FIXED: Search for model in project structure with more paths
            model_paths = [
                'mataas.h5', 
                'assets/mataas.h5', 
                './mataas.h5',
                './assets/mataas.h5',
                '../assets/mataas.h5',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mataas.h5'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'mataas.h5')
            ]
            
            # Log the search paths
            log(f"Searching for model in paths: {model_paths}")
            
            model_path = None
            for path in model_paths:
                log(f"Checking path: {path}")
                if os.path.exists(path):
                    log(f"Found model at: {path}")
                    model_path = path
                    break
            
            if not model_path:
                # Try to find model by searching directories
                log("Model not found in expected locations, searching directories...")
                
                def find_file(name, search_path):
                    for root, dirs, files in os.walk(search_path):
                        if name in files:
                            return os.path.join(root, name)
                    return None
                
                # Start search from current directory and parent
                current_dir = os.path.abspath(os.path.dirname(__file__))
                parent_dir = os.path.dirname(current_dir)
                
                # Try to find model file
                found_path = find_file('mataas.h5', current_dir)
                if not found_path:
                    found_path = find_file('mataas.h5', parent_dir)
                
                if found_path:
                    log(f"Found model by directory search: {found_path}")
                    model_path = found_path
                else:
                    log("Error: Model file not found after searching directories")
                    return False
            
            # IMPROVEMENT: Load model with performance optimizations
            log(f"Loading model from {model_path}")
            
            # Configure TensorFlow for CPU
            tf_config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=2,
                inter_op_parallelism_threads=2
            )
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))
            
            # Load model with minimal options
            self.model = load_model(model_path, compile=False)
            
            # IMPROVEMENT: Pre-warm the model with a single inference
            # This compiles any operations and prevents the first real inference from being slow
            dummy_input = np.zeros((1, self.imgSize, self.imgSize, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            log("Model warmed up with dummy inference")
            
            # FIXED: Search for labels with more paths
            # Use the model's directory as the first place to look
            model_dir = os.path.dirname(model_path)
            label_paths = [
                os.path.join(model_dir, 'mataas.json'),  # Try model directory first
                'mataas.json', 
                'assets/mataas.json', 
                './mataas.json',
                './assets/mataas.json',
                '../assets/mataas.json',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mataas.json'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'mataas.json')
            ]
            
            # Log the search paths
            log(f"Searching for labels in paths: {label_paths}")
            
            label_path = None
            for path in label_paths:
                log(f"Checking path: {path}")
                if os.path.exists(path):
                    log(f"Found labels at: {path}")
                    label_path = path
                    break
            
            if not label_path:
                # Try to find labels by searching directories
                log("Labels not found in expected locations, searching directories...")
                
                def find_file(name, search_path):
                    for root, dirs, files in os.walk(search_path):
                        if name in files:
                            return os.path.join(root, name)
                    return None
                
                # Start search from current directory and parent
                current_dir = os.path.abspath(os.path.dirname(__file__))
                parent_dir = os.path.dirname(current_dir)
                
                # Try to find labels file
                found_path = find_file('mataas.json', current_dir)
                if not found_path:
                    found_path = find_file('mataas.json', parent_dir)
                
                if found_path:
                    log(f"Found labels by directory search: {found_path}")
                    label_path = found_path
                else:
                    log("Error: Labels file not found after searching directories")
                    return False
            
            # Load labels
            import json
            with open(label_path, 'r') as f:
                self.labels = json.load(f)
            
            log(f"Labels loaded: {len(self.labels)} classes")
            return True
            
        except Exception as e:
            log(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def init_detector(self):
        """Initialize hand detector using only the fallback method to avoid protobuf errors"""
        try:
            log("Initializing hand detector...")
            
            # Skip problematic libraries entirely and use only fallback detection
            log("Using fallback skin color detection to avoid protobuf errors")
            self.use_fallback_detection = True
            
            # Initialize HSV thresholds for skin detection
            self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Initialize morphology kernel
            self.kernel = np.ones((5, 5), np.uint8)
            
            log("Fallback skin detection initialized")
            return True
            
        except Exception as e:
            log(f"Hand detector initialization error: {e}")
            # Still use fallback as a last resort
            self.use_fallback_detection = True
            return True
    
    def init_camera(self):
        """Initialize the camera focusing on index 0 only"""
        try:
            log("Opening camera with focus on index 0...")
            
            # Try with different backend options on index 0 only
            backends = [
                cv2.CAP_ANY,      # Auto-detect
                cv2.CAP_DSHOW,    # DirectShow (on Windows)
                cv2.CAP_V4L2,     # Video4Linux2 (on Linux)
                cv2.CAP_MSMF      # Media Foundation (on Windows)
            ]
            
            # Try each backend until one works
            for backend in backends:
                try:
                    log(f"Trying camera 0 with backend {backend}")
                    cap = cv2.VideoCapture(0, backend)
                    
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        for _ in range(3):  # Try a few times
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                log(f"Successfully opened camera 0 with backend {backend}")
                                
                                # Set camera properties
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                cap.set(cv2.CAP_PROP_FPS, 15)
                                
                                self.cap = cap
                                threading.Thread(target=self.camera_loop, daemon=True).start()
                                return True
                            time.sleep(0.1)
                        cap.release()  # Release if we couldn't read frames
                except Exception as e:
                    log(f"Error with camera 0, backend {backend}: {e}")
            
            # Try once more with default settings as last resort
            log("Trying camera 0 with default settings as last resort")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    log("Successfully opened camera 0 with default settings")
                    
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    self.cap = cap
                    threading.Thread(target=self.camera_loop, daemon=True).start()
                    return True
                cap.release()
            
            log("Camera initialization failed with all backends")
            
            # Continue with mock camera for testing
            log("Creating mock camera for testing")
            self.use_mock_camera = True
            threading.Thread(target=self.mock_camera_loop, daemon=True).start()
            return True
            
        except Exception as e:
            log(f"Camera initialization error: {e}")
            
            # Continue with mock camera
            log("Creating mock camera after exception")
            self.use_mock_camera = True
            threading.Thread(target=self.mock_camera_loop, daemon=True).start()
            return True
    
    def mock_camera_loop(self):
        """Provides a mock camera feed for testing"""
        log("Mock camera loop started")
        
        # Create a test pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Unavailable", (120, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Testing UI Only", (180, 280), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        while self.is_running:
            # Update with animation
            temp_frame = frame.copy()
            cv2.putText(temp_frame, time.strftime("%H:%M:%S"), (250, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            # Update the UI
            Clock.schedule_once(partial(self.update_camera_image, temp_frame.copy()), 0)
            
            # Slower refresh rate for mock camera
            time.sleep(0.5)
    
    def camera_loop(self):
        """Camera capture loop with better error handling for MSMF errors"""
        log("Camera loop started")
        
        # IMPROVEMENT: Reduce frame processing to save CPU
        frame_skip = 0
        error_count = 0
        last_successful_frame = None
        
        while self.is_running:
            if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            try:
                # Read a frame with error handling
                ret, frame = self.cap.read()
                
                # Handle frame capture errors
                if not ret or frame is None or frame.size == 0:
                    error_count += 1
                    log(f"Camera read error ({error_count})")
                    
                    # If we've had many consecutive errors, try to recover
                    if error_count > 10:
                        log("Too many camera errors, attempting to recover")
                        if self.cap:
                            self.cap.release()  # Release the old capture
                        
                        # Try to reopen with DirectShow backend
                        try:
                            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                            error_count = 0
                        except:
                            log("Failed to recover camera")
                    
                    # Use last successful frame if available, otherwise wait
                    if last_successful_frame is not None:
                        frame = last_successful_frame.copy()
                    else:
                        time.sleep(0.1)
                        continue
                else:
                    # Reset error count and save successful frame
                    error_count = 0
                    last_successful_frame = frame.copy()
                
                # Skip frames if the queue is getting full to reduce latency
                frame_skip = (frame_skip + 1) % 2  # Process every other frame
                if frame_skip != 0 and not self.frame_queue.empty():
                    continue
                
                # IMPROVEMENT: Resize frame to reduce processing overhead
                # Only if resolution is higher than needed
                if frame.shape[1] > 640 or frame.shape[0] > 480:
                    frame = cv2.resize(frame, (640, 480))
                
                # Update the camera display directly
                Clock.schedule_once(partial(self.update_camera_image, frame.copy()), 0)
                
                # Put frame in processing queue (non-blocking)
                if self.model_loaded.is_set():  # Only process if model is ready
                    try:
                        if not self.frame_queue.full():
                            self.frame_queue.put_nowait(frame)
                    except:
                        pass
                
                # Short sleep to avoid maxing CPU
                time.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                log(f"Camera loop error: {e}")
                
                # If errors persist, try to recreate the camera
                if error_count > 20:
                    log("Critical camera failure, attempting to recreate camera")
                    if hasattr(self, 'cap') and self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                    
                    # Try to initialize with another backend
                    try:
                        log("Trying to reinitialize camera with DirectShow")
                        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        error_count = 0
                    except:
                        log("Failed to recreate camera")
                
                time.sleep(0.1)
    
    def update_camera_image(self, frame, dt):
        """Updates the camera image from the UI thread"""
        # Convert to texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_image.texture = texture
    
    def processing_loop(self):
        """Processing loop with stabilization to prevent twitching"""
        log("Processing loop started with stabilized detection")
        
        # Add tracking variables to reduce twitching
        last_prediction = None
        last_confidence = 0.0
        prediction_count = 0
        required_consistent_frames = 3  # Require multiple consistent frames
        
        # For stabilization
        self.prev_hand_regions = []  # Store previous hand regions
        max_regions = 5  # Number of previous regions to store
        
        # IMPROVEMENT: Reduce logging frequency
        last_log_time = 0
        log_interval = 5  # Only log every 5 seconds
        
        # Wait until camera is ready
        while self.is_running and (not hasattr(self, 'cap') or not self.cap):
            time.sleep(0.1)
        
        log("Camera ready, beginning hand detection")
        
        while self.is_running:
            try:
                # Get a frame from the queue with timeout
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Detect hands using skin color detection
                hands = []
                hand_detected = False
                
                # Simple skin color detection (no MediaPipe or cvzone)
                try:
                    hand_region = self.detect_skin(frame)
                    
                    # Stabilize hand region using averages from previous frames
                    if hand_region is not None:
                        # Add to history
                        self.prev_hand_regions.append(hand_region)
                        # Keep only the most recent regions
                        if len(self.prev_hand_regions) > max_regions:
                            self.prev_hand_regions.pop(0)
                        
                        # Average the hand regions for stability
                        if len(self.prev_hand_regions) >= 2:
                            avg_x = sum(r[0] for r in self.prev_hand_regions) // len(self.prev_hand_regions)
                            avg_y = sum(r[1] for r in self.prev_hand_regions) // len(self.prev_hand_regions)
                            avg_w = sum(r[2] for r in self.prev_hand_regions) // len(self.prev_hand_regions)
                            avg_h = sum(r[3] for r in self.prev_hand_regions) // len(self.prev_hand_regions)
                            
                            # Use the averaged region
                            hand_region = (avg_x, avg_y, avg_w, avg_h)
                        
                        hand_detected = True
                        x, y, w, h = hand_region
                        
                        # Create a visualization frame for debugging
                        debug_frame = frame.copy()
                        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Add text showing the number of consistent predictions
                        if last_prediction:
                            cv2.putText(debug_frame, f"{last_prediction}: {prediction_count}/{required_consistent_frames}", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Update the view with the detection box
                        Clock.schedule_once(partial(self.update_camera_image, debug_frame), 0)
                        
                        # Create hand info in compatible format
                        hands = [{'bbox': (x, y, w, h)}]
                    else:
                        # Reset stabilization when no hand detected
                        self.prev_hand_regions = []
                        prediction_count = 0
                        
                except Exception as e:
                    log(f"Skin detection error: {e}")
                
                # Update detection status (via kivy Clock to be thread-safe)
                Clock.schedule_once(lambda dt: self.update_detection_status(hand_detected), 0)
                
                # Process hand for prediction if needed
                current_time = time.time()
                if hand_detected:
                    # Get hand and make prediction
                    hand = hands[0]
                    
                    # Make prediction
                    letter, conf = self.predict_letter(frame, hand, False)
                    
                    # Stability check - require multiple consistent predictions
                    if letter and conf > 0.5:
                        if letter == last_prediction:
                            prediction_count += 1
                        else:
                            # New prediction - reset counter
                            last_prediction = letter
                            prediction_count = 1
                            
                        # Log current tracking state occasionally
                        if (current_time - last_log_time) >= log_interval:
                            log(f"Tracking: {letter} ({conf:.2f}) - {prediction_count}/{required_consistent_frames}")
                            last_log_time = current_time
                        
                        # Only update UI when we have enough consistent frames
                        if prediction_count >= required_consistent_frames:
                            if current_time - self.last_prediction_time >= self.prediction_interval:
                                log(f"Consistent prediction: {letter} with confidence {conf:.2f}")
                                self.last_prediction_time = current_time
                                
                                # Put result in queue
                                self.result_queue.put((letter, conf))
                                
                                # Reset counter to prevent rapid repeat predictions
                                prediction_count = 0
                else:
                    # Reset tracking when no hand detected
                    last_prediction = None
                    prediction_count = 0
                
                # Check for results
                self.check_results()
                
                # Short sleep to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                log(f"Processing error: {e}")
                time.sleep(0.1)
    
    def detect_skin(self, frame):
        """Improved skin color detection as the primary hand detection method"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color (use class variables for consistency)
            lower_skin = self.lower_skin
            upper_skin = self.upper_skin
            
            # Create a mask
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            mask = cv2.dilate(mask, self.kernel, iterations=2)
            mask = cv2.erode(mask, self.kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (likely the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Minimum size check
                area = cv2.contourArea(max_contour)
                if area > 3000:  # Adjust threshold as needed
                    x, y, w, h = cv2.boundingRect(max_contour)
                    
                    # Make sure the region is reasonable for a hand
                    if w > 20 and h > 20 and w < frame.shape[1]*0.8 and h < frame.shape[0]*0.8:
                        # Expand the bounding box slightly
                        padding = 20
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + padding*2)
                        h = min(frame.shape[0] - y, h + padding*2)
                        
                        return (x, y, w, h)
            
            return None
            
        except Exception as e:
            log(f"Skin detection error: {e}")
            return None
            log(f"Processing error: {e}")
            time.sleep(0.1)
    
    def update_detection_status(self, detected):
        """Update hand detection status (called from main thread)"""
        self.hand_detected = detected
        self.status_color = [0, 1, 0, 1] if detected else [1, 0, 0, 1]
    
    def check_results(self):
        """Check for prediction results with error handling"""
        try:
            # IMPROVEMENT: Process all results at once
            results = []
            while not self.result_queue.empty():
                try:
                    results.append(self.result_queue.get_nowait())
                except:
                    break
            
            if results:
                # Update UI with the most confident prediction
                try:
                    results.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
                    letter, conf = results[0]
                    
                    # Validate values before updating UI
                    if isinstance(letter, str) and letter:
                        if isinstance(conf, (int, float)) and not math.isnan(conf) and not math.isinf(conf):
                            # Schedule the update with validated values
                            Clock.schedule_once(lambda dt: self.update_prediction(letter, float(conf)), 0)
                except Exception as e:
                    log(f"Error processing results: {e}")
        except Exception as e:
            log(f"Error in check_results: {e}")
            # Continue without crashing
    
    def update_prediction(self, letter, conf):
        """Update prediction UI (called from main thread) with error handling for confidence value"""
        try:
            self.current_letter = letter
            
            # Ensure confidence is a valid float between 0.0 and 1.0
            try:
                # Convert to float and ensure valid range
                conf_value = float(conf)
                if math.isnan(conf_value) or math.isinf(conf_value):
                    conf_value = 0.0
                conf_value = max(0.0, min(1.0, conf_value))
                self.confidence = conf_value
            except Exception as e:
                log(f"Error converting confidence value: {e}")
                self.confidence = 0.0
            
            # Add letter to current word
            self.current_word.append(letter)
            self.translated_word = "".join(self.current_word)
            log(f"Updated prediction: {letter} with confidence {self.confidence:.2f}, word: {self.translated_word}")
        except Exception as e:
            log(f"Error in update_prediction: {e}")
            # Continue without crashing
    
    def predict_letter(self, frame, hand, should_log=False):
        """Predict the letter from a hand region with better error handling"""
        if not self.model or not self.labels:
            if should_log:
                log("Model or labels not available")
            return None, 0.0
        
        try:
            # Extract bounding box
            x, y, w, h = hand['bbox']
            
            # Ensure valid bounding box values
            if not all(isinstance(val, (int, float)) for val in [x, y, w, h]):
                log(f"Invalid bbox values: {hand['bbox']}")
                return None, 0.0
            
            # Convert to integers and ensure positive values
            x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
            
            # IMPROVEMENT: Add extra padding for better predictions
            offset = self.offset
            
            # Crop with offset
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(frame.shape[1], x + w + offset)
            y2 = min(frame.shape[0], y + h + offset)
            
            # Check for valid dimensions before cropping
            if x1 >= x2 or y1 >= y2:
                if should_log:
                    log("Invalid crop dimensions")
                return None, 0.0
            
            # Crop the hand region
            imgCrop = frame[y1:y2, x1:x2]
            
            # Check if crop is valid
            if imgCrop.size == 0:
                if should_log:
                    log("Empty crop region")
                return None, 0.0
            
            # IMPROVEMENT: Save debug images less frequently
            if should_log:
                timestamp = int(time.time())
                cv2.imwrite(f"hand_crop_{timestamp}.jpg", imgCrop)
            
            # Create white background
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            
            # Resize and center hand on white background
            try:
                # Calculate aspect ratio safely
                aspectRatio = float(h) / max(1, float(w))  # Avoid division by zero
                
                if aspectRatio > 1:
                    # Hand is taller than wide
                    k = self.imgSize / float(h)
                    wCal = max(1, math.ceil(k * w))  # Ensure at least 1 pixel
                    
                    if wCal > 0 and self.imgSize > 0:
                        # IMPROVEMENT: Use INTER_NEAREST for faster resizing
                        imgResize = cv2.resize(imgCrop, (wCal, self.imgSize), 
                                            interpolation=cv2.INTER_NEAREST)
                        wGap = math.ceil((self.imgSize - wCal) / 2)
                        
                        # Check for valid dimensions before copying
                        if wGap >= 0 and wGap + imgResize.shape[1] <= self.imgSize:
                            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    # Hand is wider than tall
                    k = self.imgSize / float(w)
                    hCal = max(1, math.ceil(k * h))  # Ensure at least 1 pixel
                    
                    if hCal > 0 and self.imgSize > 0:
                        # IMPROVEMENT: Use INTER_NEAREST for faster resizing
                        imgResize = cv2.resize(imgCrop, (self.imgSize, hCal),
                                            interpolation=cv2.INTER_NEAREST)
                        hGap = math.ceil((self.imgSize - hCal) / 2)
                        
                        # Check for valid dimensions before copying
                        if hGap >= 0 and hGap + imgResize.shape[0] <= self.imgSize:
                            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize
            except Exception as e:
                if should_log:
                    log(f"Resize error: {e}")
                return None, 0.0
            
            # IMPROVEMENT: Save processed images less frequently
            if should_log:
                cv2.imwrite(f"hand_processed_{timestamp}.jpg", imgWhite)
            
            # Prepare for prediction
            # IMPROVEMENT: Normalize directly without copying
            img = imgWhite.astype("float32") / 255.0
            
            # Make prediction with error handling
            try:
                # IMPROVEMENT: Reduce verbosity
                prediction = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                
                # Ensure prediction is valid
                if prediction is None or len(prediction) == 0:
                    return None, 0.0
                
                # Get highest confidence class
                index = int(np.argmax(prediction))
                if index < 0 or index >= len(self.labels):
                    return None, 0.0
                    
                letter = self.labels[index]
                confidence = float(prediction[index])
                
                # Ensure valid confidence value
                if math.isnan(confidence) or math.isinf(confidence):
                    confidence = 0.0
                
                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, confidence))
                
                return letter, confidence
            except Exception as e:
                if should_log:
                    log(f"Prediction error: {e}")
                return None, 0.0
            
        except Exception as e:
            if should_log:
                log(f"Hand processing error: {e}")
            return None, 0.0
    
    def update_display(self, dt):
        """Update the display (called by Clock)"""
        # No longer needed as we update the display directly from the camera thread
        pass
    
    def delete_last_letter(self):
        """Delete the last letter from the current word"""
        if self.current_word:
            deleted = self.current_word.pop()
            self.translated_word = "".join(self.current_word)
            log(f"Deleted last letter: {deleted}, word now: {self.translated_word}")
    
    def clear_word(self):
        """Clear the entire word"""
        old_word = "".join(self.current_word)
        self.current_word.clear()
        self.translated_word = ""
        log(f"Cleared word: {old_word}")
    
    def on_stop(self):
        """Clean up resources when the app is closed"""
        log("Application stopping - cleaning up resources")
        
        # Signal threads to stop
        self.is_running = False
        
        # Give threads time to exit
        time.sleep(0.1)
        
        # Release camera
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            log("Camera released")
        
        # Release MediaPipe resources if used
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            log("MediaPipe resources released")
        
        # Close log file
        log("Application cleanup complete")

# IMPROVEMENT: Define UI in code to avoid loading a .kv file
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

class ASLTranslatorApp(App):
    """Main application class"""
    
    def build(self):
        """Build the application"""
        self.title = 'ASL Translator'
        
        # Create the main layout
        main_layout = BoxLayout(orientation='vertical', padding=[10, 10, 10, 10], spacing=10)
        
        # Title label
        title = Label(
            text='ASL Translator',
            font_size='24sp',
            size_hint=(1, None),
            height=40
        )
        main_layout.add_widget(title)
        
        # Subtitle label
        subtitle = Label(
            text='Translate American Sign Language in real-time',
            font_size='14sp',
            size_hint=(1, None),
            height=30
        )
        main_layout.add_widget(subtitle)
        
        # Create the camera container
        camera_container = BoxLayout(size_hint=(1, 0.7))
        main_layout.add_widget(camera_container)
        
        # Create status indicator box
        status_box = BoxLayout(
            size_hint=(1, None),
            height=30,
            spacing=10
        )
        
        # Create status indicator widget
        from kivy.uix.widget import Widget
        status_indicator = Widget(
            size_hint=(None, None),
            size=(20, 20)
        )
        status_box.add_widget(status_indicator)
        
        # Status label
        status_label = Label(
            text='No hand detected',
            size_hint=(1, None),
            height=30
        )
        status_box.add_widget(status_label)
        main_layout.add_widget(status_box)
        
        # Translation label
        translation_label = Label(
            text='Translation',
            size_hint=(1, None),
            height=30
        )
        main_layout.add_widget(translation_label)
        
        # Translation text input
        translation_text = TextInput(
            size_hint=(1, None),
            height=70,
            readonly=True,
            multiline=True
        )
        main_layout.add_widget(translation_text)
        
        # Button box
        button_box = BoxLayout(
            size_hint=(1, None),
            height=50,
            spacing=10
        )
        
        # Delete button
        delete_button = Button(
            text='Delete Last Letter',
            size_hint=(0.5, 1)
        )
        button_box.add_widget(delete_button)
        
        # Clear button
        clear_button = Button(
            text='Clear All',
            size_hint=(0.5, 1)
        )
        button_box.add_widget(clear_button)
        main_layout.add_widget(button_box)
        
        # Version label
        version_label = Label(
            text='ASL Translator v1.0 | SallInterpret',
            font_size='12sp',
            size_hint=(1, None),
            height=20
        )
        main_layout.add_widget(version_label)
        
        # Create translator widget with custom UI
        translator = ASLTranslatorWidget()
        
        # Store references to widgets
        translator.ids = {
            'camera_container': camera_container,
            'status_indicator': status_indicator,
            'status_label': status_label,
            'translation_text': translation_text
        }
        
        # Bind properties
        translator.bind(
            hand_detected=lambda obj, val: setattr(
                status_label, 'text', 'Hand detected' if val else 'No hand detected'
            ),
            translated_word=lambda obj, val: setattr(translation_text, 'text', val),
            status_color=lambda obj, val: self.update_status_color(status_indicator, val)
        )
        
        # Bind buttons
        delete_button.bind(on_press=lambda x: translator.delete_last_letter())
        clear_button.bind(on_press=lambda x: translator.clear_word())
        
        return translator
    
    def update_status_color(self, widget, value):
        """Updates the status indicator color"""
        from kivy.graphics import Color, Ellipse
        widget.canvas.clear()
        with widget.canvas:
            Color(*value)
            Ellipse(pos=widget.pos, size=widget.size)
    
    def on_stop(self):
        """Handle application stop"""
        if hasattr(self.root, 'on_stop'):
            self.root.on_stop()

if __name__ == '__main__':
    try:
        # Log the current working directory and available files to help debugging
        current_dir = os.path.abspath(os.path.dirname(__file__))
        print(f"Current working directory: {current_dir}")
        print(f"Files in current directory: {os.listdir(current_dir)}")
        
        # Try to list assets directory if it exists
        assets_dir = os.path.join(current_dir, 'assets')
        if os.path.exists(assets_dir) and os.path.isdir(assets_dir):
            print(f"Files in assets directory: {os.listdir(assets_dir)}")
        else:
            print("Assets directory not found in current location")
            
        # Check parent directory assets
        parent_dir = os.path.dirname(current_dir)
        parent_assets = os.path.join(parent_dir, 'assets')
        if os.path.exists(parent_assets) and os.path.isdir(parent_assets):
            print(f"Files in parent assets directory: {os.listdir(parent_assets)}")
        
        # IMPROVEMENT: Set environment variables at the start
        log("Starting ASL Translator application")
        ASLTranslatorApp().run()
        log("Application closed normally")
    except Exception as e:
        log(f"Application error: {e}")
        import traceback
        traceback.print_exc()