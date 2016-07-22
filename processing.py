import numpy as np
import scipy as sp
import scipy.signal
import cv2

def active_show(name, image):
    cv2.imshow(name, image)
    cv2.imwrite(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def no_show(name, image):
    # No opt
    pass


def join_image(image_1, image_2, scale=0.5, margin=0):

    joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                             image_1.shape[1] + image_2.shape[1] + margin,
                             3), dtype=image_1.dtype)

    joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
    joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

    return cv2.resize(joined_image, (0, 0), fx=scale, fy=scale)


def detect_players(shot_image, show=active_show):
    global hog

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(shot_image, winStride=(1, 1),
                                            padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(shot_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    show('detected_players', shot_image)



def canny_contours(moving_bits, show=active_show):
    tmp = moving_bits.copy()
    tmp2 = moving_bits.copy()
    out = np.zeros(moving_bits.shape, dtype=moving_bits.dtype)
    show('/tmp/canny_input.png', moving_bits)

    reduced = cv2.Canny(moving_bits, 100, 200)
    show('/tmp/canny_edge.png', reduced)

    imgray = reduced
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    show('/tmp/canny_thresh.png', thresh)
    im2, contours, hierarchy = cv2.findContours(reduced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(tmp, contours, -1, (255, 0, 0))
    show('/tmp/canny_contours.png', tmp)

    #contours = sorted(contours, key=lambda x: cv2.contourArea(x))

    # > 10 < 30 worked for 0052
    rects = [ cv2.boundingRect(x) for x in contours if len(x) < 30 ]
    for r in rects:
        (x, y, w, h) = r
        if w*h < 200:
            cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), -1)
            cv2.rectangle(tmp2, (x, y), (x+w, y+h), (255, 0, 0), -1)

    #out = cv2.dilate(out, None, iterations=10)

    show('/tmp/canny_out2.png', tmp2)
    show('/tmp/canny_out.png', out)

    #contours = [ x for x in contours if cv2.contourArea(x) > 0 and cv2.contourArea(x) < 30 ]


    #cv2.drawContours(tmp, contours, -1, (0, 255, 0), 5)

    #cv2.imshow('tmp', tmp)
    #cv2.waitKey(0)
    #cv2.drawContours(reduced, contours, len(contours) - 10, (0, 255, 0), 3)

    #    for c in contours:
    #        cv2.fillConvexPoly(red uced, c, (0, 255, 0))

    #cv2.imshow('tmp', tmp)
    #cv2.waitKey(0)

    return out

    imgray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100,
                               param2=10,
                               minRadius=0,
                               maxRadius=10)

    show('tmp', tmp)


    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(out, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(out, (i[0], i[1]), 2, (0, 0, 255), 3)

    #    cv2.imshow('tmp', tmp)
    #    cv2.waitKey(0)

    show('out', out)

def blob_detection(moving_bits, show=active_show):
    out = np.zeros(moving_bits.shape, dtype=moving_bits.dtype)
    tmp = moving_bits.copy()
    detector = blob_detector()

    keypoints = detector.detect(moving_bits)

    show('/tmp/blob_input.png', moving_bits)

    sq = 5
    for kp in keypoints:
        (x, y) = kp.pt
        cv2.rectangle(out, (int(x - sq), int(y - sq)), (int(x + sq), int(y + sq)), (0, 255, 0), -1)
        cv2.rectangle(tmp, (int(x - sq), int(y - sq)), (int(x + sq), int(y + sq)), (0, 255, 0), -1)


    show('/tmp/blob_kp.png', tmp)
    show('/tmp/blob_output.png', out)

    #moving_bits = cv2.drawKeypoints(moving_bits, keypoints, out, (255, 255, 0),
    #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #show('moving', moving_bits)

    #show('out', out)

    return out
    #show('out', out)

    #return out


def blob_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 50
    params.maxThreshold = 200

    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 200

    params.filterByCircularity = True
    params.minCircularity = 0.70

    params.filterByConvexity = True
    params.minConvexity = 0.40

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    return cv2.SimpleBlobDetector_create(params)

def moving_parts(prev_image, shot_image, show=active_show):
    last_frame = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    this_frame = cv2.cvtColor(shot_image, cv2.COLOR_BGR2GRAY)

    show('/tmp/last_frame.png', last_frame)
    show('/tmp/this_frame.png', this_frame)

    frame_delta = cv2.absdiff(last_frame, this_frame)

    #frame_delta = cv2.dilate(frame_delta, None, iterations=10)
    #frame_delta = cv2.erode(frame_delta, None, iterations=1)
    threshold = frame_delta

    show('/tmp/frame_delta.png', frame_delta)

    (_, threshold) = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)


    mask = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    show('/tmp/mask.png', mask)
    mask = cv2.dilate(mask, None, iterations=10)

    show('/tmp/dilated_mask.png', mask)

    # Ignore the top 20% and the bottom 20% of the screen. Puck is probably not here
    top_y = int(mask.shape[0] * .20)
    bottom_y = int(mask.shape[0] * .80)
    mask[0:top_y,:] = 0
    mask[bottom_y:mask.shape[0],:] = 0

    moving_bits = np.bitwise_and(shot_image, mask)
    #moving_bits = np.bitwise_and(reduce_image(shot_image), mask)

    show('/tmp/moving_bits.png', moving_bits)

    white_to_black_mask = mask[:,:,:] <= 0

    whited_moving_bits = np.copy(moving_bits)
    whited_moving_bits[white_to_black_mask] = 255

    show('/tmp/whited.png', whited_moving_bits)

    return (frame_delta, moving_bits, whited_moving_bits)


def reduced_contours(img, show=active_show):
    tmp = img.copy()
    tmp2 = img.copy()

    out = np.zeros(img.shape, dtype=img.dtype)
    black_lower = (0, 0, 0)
    black_upper = (120, 120, 120)

    show('/tmp/original.png', img)
    mask = cv2.inRange(img, black_lower, black_upper)
    show('/tmp/black_mask.png', mask)

    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    show('/tmp/dilated_mask.png', mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    (ret, cnts, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(tmp2, cnts, -1, (0, 0, 255))
    show('/tmp/all_contours.png', tmp2)

    #contours = cnts
    contours = [ x for x in cnts if cv2.contourArea(x) > 0 and cv2.contourArea(x) < 100 ]

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(out, (x,y), (x+w, y+h), (0, 0, 255), -1)
        cv2.rectangle(tmp, (x,y), (x+w, y+h), (0, 0, 255), -1)

    show('/tmp/contours_overlay.png', tmp)
    show('/tmp/contours.png', out)

    return out


def reduce_image(img, show=active_show):
    out = np.zeros(img.shape, dtype=img.dtype)
    black_lower = (0, 0, 0)
    black_upper = (90, 90, 90)

    #show('original', img)
    mask = cv2.inRange(img, black_lower, black_upper)
    #show('black mask', mask)

    #mask = cv2.dilate(mask, None, iterations=2)
    #mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return np.bitwise_and(img, mask)


