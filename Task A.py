import numpy as np
import cv2

def blur_faces(frame, faces):
    # Apply Gaussian blur to faces with a specific kernel size
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi, (75, 75), 0)
        frame[y:y+h, x:x+w] = blur

def increase_brightness(frame):
    # Increase brightness if the average brightness is below a threshold
    avg = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    night_thresh = 100
    if avg < night_thresh:
        increase = 30
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, increase)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame

def overlay_talking(frame, talking, x_offset, y_offset):
    # Overlay talking video on the frame with a black border
    talking_resize = cv2.resize(talking, (320, 180))
    border_size = 5
    border_color = (0, 0, 0)
    talking_with_border = cv2.copyMakeBorder(talking_resize, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    frame[y_offset: y_offset + talking_with_border.shape[0], x_offset: x_offset + talking_with_border.shape[1]] = talking_with_border

def watermark(frame, watermark1, watermark2, alpha, frame_count):
    # Apply watermark based on frame count, alternating between two watermarks
    if frame_count % 150 < 100:
        return cv2.addWeighted(frame, 1, watermark1, alpha, 0)
    else:
        return cv2.addWeighted(frame, 1, watermark2, alpha, 0)

def process_video(input_video_path, talking_path, face_cascade_path, watermark1, 
                  watermark2, endscreen_path, output_video_path):
    # Open video capture objects
    vid = cv2.VideoCapture(input_video_path)
    talking = cv2.VideoCapture(talking_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Get total number of frames in the video
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_no_frames <= 0:
        print(f"Error: Failed to get the frame count for {input_video_path}.")
        return

    processed_frames = []

    for frame_count in range(total_no_frames):
        success_img, img = vid.read()
        success_talk, talk = talking.read()

        if not success_img or not success_talk:
            # Handle errors
            break

        # Face blurring
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        blur_faces(img, faces)

        # Increase brightness if it's nighttime
        img = increase_brightness(img)

        # Talking overlay
        overlay_talking(img, talk, x_offset=30, y_offset=30)

        # Watermark
        alpha = 0.5
        img = watermark(img, watermark1, watermark2, alpha, frame_count)

        processed_frames.append(img)

    # Release VideoCapture objects
    vid.release()
    talking.release()
    
    # Write processed frames to output video
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 
                                   30.0, (1280, 720), isColor=True)
    for frame in processed_frames:
        output_video.write(frame)
    
    # Add end screen video
    endscreen = cv2.VideoCapture(endscreen_path)
    while True:
        success_endscreen, endscreen_frame = endscreen.read()
        if not success_endscreen:
            break
        output_video.write(endscreen_frame)
    # Release VideoCapture object for the end screen
    endscreen.release()
        # Set the frame count
    output_video.set(cv2.CAP_PROP_FRAME_COUNT, total_no_frames)
        # Release VideoWriter
    output_video.release()
    
    # User confirmation message
    print(f"Processing complete for {input_video_path}. Output video saved as {output_video_path}.")
    print()

# Define watermark images
watermark1 = cv2.imread("watermark1.png")
watermark2 = cv2.imread("watermark2.png")

# Process each video separately
process_video("alley.mp4", "talking.mp4", "face_detector.xml", watermark1, watermark2, "endscreen.mp4", "output_office.avi")
process_video("office.mp4", "talking.mp4", "face_detector.xml", watermark1, watermark2, "endscreen.mp4", "output_alley.avi")
process_video("singapore.mp4", "talking.mp4", "face_detector.xml", watermark1, watermark2, "endscreen.mp4", "output_singapore.avi")
process_video("traffic.mp4", "talking.mp4", "face_detector.xml", watermark1, watermark2, "endscreen.mp4", "output_traffic.avi")

# Confirmation message
print('All outputs generated. Please check your output folder.')

cv2.destroyAllWindows()
