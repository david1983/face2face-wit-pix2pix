import os
import cv2
import dlib
import time
import argparse
import numpy as np
import sys
from imutils import video

DOWNSAMPLE_RATIO = 1


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)



def main():
    os.makedirs('original', exist_ok=True)
    os.makedirs('landmarks', exist_ok=True)
    try:
        cap = cv2.VideoCapture(args.filename)
    except:
        print("error")
    fps = video.FPS().start()
    

    print("starting")
    print(cap.isOpened())
    count = 0
    prevt = time.time()
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
        
        prevt = time.time()
        (height, width, channels) = frame_raw.shape
        frame = frame_raw[0:height-200, 200:width-200]
        frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
       
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        black_image = np.zeros(frame.shape, np.uint8)

        t = time.time()
        (x, y, w, h) = (0,0,0,0)
        # Perform if there is a face detected
        if len(faces) == 1:
            for face in faces:
                (x, y, w, h) = rect_to_bb(face)
                detected_landmarks = predictor(gray, face).parts()
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])
                left_eyebrow = reshape_for_polyline(landmarks[22:27])
                right_eyebrow = reshape_for_polyline(landmarks[17:22])
                nose_bridge = reshape_for_polyline(landmarks[27:31])
                lower_nose = reshape_for_polyline(landmarks[30:35])
                left_eye = reshape_for_polyline(landmarks[42:48])
                right_eye = reshape_for_polyline(landmarks[36:42])
                outer_lip = reshape_for_polyline(landmarks[48:60])
                inner_lip = reshape_for_polyline(landmarks[60:68])

                color = (255, 255, 255)
                thickness = 3

                cv2.polylines(black_image, [jaw], False, color, thickness)
                cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
                cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
                cv2.polylines(black_image, [nose_bridge], False, color, thickness)
                cv2.polylines(black_image, [lower_nose], True, color, thickness)
                cv2.polylines(black_image, [left_eye], True, color, thickness)
                cv2.polylines(black_image, [right_eye], True, color, thickness)
                cv2.polylines(black_image, [outer_lip], True, color, thickness)
                cv2.polylines(black_image, [inner_lip], True, color, thickness)



            padd = int(80 / DOWNSAMPLE_RATIO)
            cropo = frame[
                (y-padd)*DOWNSAMPLE_RATIO:(y)*DOWNSAMPLE_RATIO+(padd+h)*DOWNSAMPLE_RATIO, 
                (x-padd)*DOWNSAMPLE_RATIO:(x)*DOWNSAMPLE_RATIO+(padd+w)*DOWNSAMPLE_RATIO]
            cropl = black_image[
                (y-padd)*DOWNSAMPLE_RATIO:(y)*DOWNSAMPLE_RATIO+(padd+h)*DOWNSAMPLE_RATIO, 
                (x-padd)*DOWNSAMPLE_RATIO:(x)*DOWNSAMPLE_RATIO+(padd+w)*DOWNSAMPLE_RATIO] 
            if(y-padd <= 0 or x-padd<=0):
                continue 
            crop_original = cropo
            crop_landmarks = cropl
            cv2.imwrite("original/{}.png".format(count+1), crop_original)
            cv2.imwrite("landmarks/{}.png".format(count+1), crop_landmarks)
            # Display the resulting frame


            padd = int(140 / DOWNSAMPLE_RATIO)
            # print(count)
            if(y-padd <= 0 or x-padd<=0):
                continue 
            
            cropo = frame[
                (y-padd)*DOWNSAMPLE_RATIO:(y)*DOWNSAMPLE_RATIO+(padd+h)*DOWNSAMPLE_RATIO, 
                (x-padd)*DOWNSAMPLE_RATIO:(x)*DOWNSAMPLE_RATIO+(padd+w)*DOWNSAMPLE_RATIO]
            cropl = black_image[
                (y-padd)*DOWNSAMPLE_RATIO:(y)*DOWNSAMPLE_RATIO+(padd+h)*DOWNSAMPLE_RATIO, 
                (x-padd)*DOWNSAMPLE_RATIO:(x)*DOWNSAMPLE_RATIO+(padd+w)*DOWNSAMPLE_RATIO] 
            cv2.imwrite("original/{}.png".format(count),  cropo )
            cv2.imwrite("landmarks/{}.png".format(count), cropl )
            fps.update()
            progress(count, args.number)
            # print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
            cv2.imshow("image", np.concatenate([cropo,cropl]))
            count += 2
            if count == args.number:  # only take 400 photos
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='filename', type=str, help='Name of the video file.')
    parser.add_argument('--num', dest='number', type=int, help='Number of train data to be created.')
    parser.add_argument('--landmark-model', dest='face_landmark_shape_file', type=str, help='Face landmark model file.')
    args = parser.parse_args()

    # Create the face predictor and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.face_landmark_shape_file)

    main()
