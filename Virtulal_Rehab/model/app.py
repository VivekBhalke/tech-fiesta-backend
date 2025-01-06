# from flask import Flask, request
# import cv2
# import numpy as np
# import os
# import shutil

# app = Flask(__name__)

# # Directory to save frames
# FRAME_DIR = "frames"
# os.makedirs(FRAME_DIR, exist_ok=True)
# def clear_frames():
#     """Remove all files in the frames directory when a new session starts."""
#     if os.path.exists(FRAME_DIR):
#         shutil.rmtree(FRAME_DIR)  # Delete the entire directory
#         os.makedirs(FRAME_DIR, exist_ok=True)  # Recreate the directory

# @app.route("/start-session", methods=["POST"])
# def start_session():
#     """Clear old frames when a new streaming session begins."""
#     clear_frames()
#     print("New session started, cleared old frames.")
#     return "Session started", 200
# @app.route("/process-frame", methods=["POST"])
# def process_frame():
#     try:
        

#         # Decode the frame
#         np_img = np.frombuffer(request.data, np.uint8)
#         frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#         # Save the frame
#         frame_path = os.path.join(FRAME_DIR, f"frame_{len(os.listdir(FRAME_DIR)) + 1}.jpg")
#         cv2.imwrite(frame_path, frame)
#         print(f"Frame saved: {frame_path}")

#         return "Frame processed", 200
#     except Exception as e:
#         print("Error processing frame:", e)
#         return "Error processing frame", 500

# if __name__ == "__main__":
#     app.run(port=8000)
# from flask import Flask, request, Response
# import cv2
# import numpy as np

# app = Flask(__name__)

# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     # Read the image from the request
#     image_np = np.frombuffer(request.data, np.uint8)
#     frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#     # Process the frame (Example: Convert to Grayscale)
#     processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Encode the processed frame back to JPEG
#     _, buffer = cv2.imencode(".jpg", processed_frame)
    
#     return Response(buffer.tobytes(), mimetype="image/jpeg")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)

from flask import Flask, request , jsonify
import cv2
import numpy as np
import os
import shutil
from squat_detect import process_frames  # Import the function

app = Flask(__name__)

# Directory to save frames
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

def clear_frames():
    """Remove all files in the frames directory when a new session starts."""
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
        os.makedirs(FRAME_DIR, exist_ok=True)

@app.route("/start-session", methods=["POST"])
def start_session():
    """Clear old frames when a new streaming session begins."""
    clear_frames()
    print("New session started, cleared old frames.")
    return "Session started", 200

@app.route("/process-frame", methods=["POST"])
def process_frame():
    try:
        # Decode the frame
        np_img = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Save the frame
        # frame_path = os.path.join(FRAME_DIR, f"frame_{len(os.listdir(FRAME_DIR)) + 1}.jpg")
        frame_path = os.path.join(FRAME_DIR, f"frame_.jpg")

        cv2.imwrite(frame_path, frame)
        print(f"Frame saved: {frame_path}")

        # Call the processing function after frame is received
        result_list =process_frames()
        print("Result list is")
        print(result_list)
        return jsonify(result_list), 200
    except Exception as e:
        print("Error processing frame:", e)
        return "Error processing frame", 500

if __name__ == "__main__":
    app.run(port=8000)
