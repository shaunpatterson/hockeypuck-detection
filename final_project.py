# Final Project
# Hockey Puck Tracker
# Shaun Patterson

# Resources
# http://stackoverflow.com/questions/8076889/tutorial-on-opencv-simpleblobdetector

import os
import numpy as np
import scipy as sp
import scipy.signal
import cv2
import argparse
import imutils

from processing import *

from imutils.object_detection import non_max_suppression

from cs6475 import *

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                                 % cv2.__version__)

hog = None


def main():
    global hog
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    ap = argparse.ArgumentParser()
    ap.add_argument("-pi", "--previmage", help="path to an image")
    ap.add_argument("-i", "--image", help="path to an image")
    ap.add_argument("-v", "--video", help="path to a video")

    args = vars(ap.parse_args())

    image = args.get('image')
    prev_image = args.get('previmage')
    video = args.get('video')

    if video:
        process_video(video)
    else:
        process_frame(cv2.imread(prev_image), cv2.imread(image), None)


def process_frame(prev_image, current_image, previous_votes, show=active_show):
    if previous_votes == None:
        previous_votes = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=current_image.dtype)

    out = np.zeros(current_image.shape, dtype=current_image.dtype)
    this_frame = current_image.copy()
    last_frame = prev_image.copy()

    # Ignore the top 20% and the bottom 20% of the screen. Puck is probably not here
    top_y = int(this_frame.shape[0] * .20)
    bottom_y = int(this_frame.shape[0] * .80)
    this_frame[0:top_y,:] = 0
    this_frame[bottom_y:this_frame.shape[0],:] = 0
    last_frame[0:top_y,:] = 0
    last_frame[bottom_y:last_frame.shape[0],:] = 0

    (frame_delta, moving, whited_moving) = moving_parts(prev_image, current_image, show)

    reduced_moving = reduce_image(whited_moving)


    contour_votes = np.zeros(whited_moving.shape, dtype=whited_moving.dtype)
    canny_votes = np.zeros(whited_moving.shape, dtype=whited_moving.dtype)
    blob_votes = np.zeros(whited_moving.shape, dtype=whited_moving.dtype)

    contour_votes = reduced_contours(whited_moving, show)
    canny_votes = canny_contours(whited_moving, show)
    blob_votes = blob_detection(reduced_moving, show)

    votes = contour_votes + canny_votes + blob_votes
    show('/tmp/votes.png', votes)
    gray_votes = cv2.cvtColor(votes, cv2.COLOR_BGR2GRAY) + previous_votes

    show('/tmp/gray_votes.png', gray_votes)

    max_value = np.max(gray_votes)
    #max_votes = gray_votes
    max_value = 255
    (ret, max_votes) = cv2.threshold(gray_votes, max_value - 1, 255, cv2.THRESH_BINARY)
    show('/tmp/max_votes.png', votes)


    dst = current_image

    (_, contours, __) = cv2.findContours(max_votes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst, contours, -1, (0, 255, 0), -1)


    # Add the votes on top of the image
    #roi = np.copy(current_image)
    #mask_inv = cv2.bitwise_not(max_votes)
    #img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    #img2_fg = cv2.bitwise_and(votes, votes, mask=max_votes)
    #dst = cv2.add(img1_bg, img2_fg)

    #alpha = .5
    #dst = cv2.addWeighted(dst, alpha, current_image, 1 - alpha, 0.0)

    (ret, reduced_max_votes) = cv2.threshold(gray_votes, max_value - 1, 1, cv2.THRESH_BINARY)

    show('/tmp/final.png', dst)

    return (dst, reduced_max_votes)




def process_video(video_filename):
    camera = cv2.VideoCapture(video_filename)

    last_frame = None
    last_votes = None
    frame_index = 0
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=800)

        if last_frame is None:
            last_frame = frame
            continue

        (output_frame, last_votes) = process_frame(last_frame, frame, last_votes, no_show)

        cv2.putText(output_frame, "Frame %s" % (frame_index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        cv2.imshow('feed', output_frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break

        last_frame = frame
        frame_index += 1

    camera.release()
    cv2.destroyAllWindows()






if __name__ == '__main__':
    main()



