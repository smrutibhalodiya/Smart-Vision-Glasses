# !usr/bin/env python

import glob
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import cv2

from utils import *

MIN_MATCH_COUNT = 5
KERNEL_SIZE = 11

currency_dir = 'currency_images'

def main():
    print('Currency Recognition Program starting...\n')

    training_set = [
        img for img in glob.glob(os.path.join(currency_dir, "*.jpg"))
    ]
    training_set_name = [
        Path(img_path).stem for img_path in training_set
    ]

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Initiate camera capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # preprocess frame
        frame = preprocess(frame)

        # keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(frame, mask=None)

        max_matches = -1
        sum_good_matches = 0
        kp_perc = 0

        for i in range(len(training_set)):
            train_img = cv2.imread(training_set[i])
            train_img = preprocess(train_img, showImages=False)
            kp2, des2 = orb.detectAndCompute(train_img, mask=None)

            # brute force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            # Match descriptors
            all_matches = bf.knnMatch(des1, des2, k=2)

            good = []

            # store all the good matches as per Lowe's ratio test.
            for m, n in all_matches:
                if m.distance < 0.6 * n.distance:
                    good.append([m])

            num_matches = len(good)
            sum_good_matches += num_matches

            if num_matches > max_matches:
                max_matches = num_matches
                best_i = i
                best_kp = kp2
                max_good_matches = len(good)
                best_img = train_img

        kp_perc = (max_good_matches / sum_good_matches * 100) if sum_good_matches > 0 else 0

        if max_matches >= MIN_MATCH_COUNT and (kp_perc >= 40):
            print(f'\nMatch Found!\n{training_set_name[best_i]} has maximum matches of {max_matches} ({kp_perc}%)')

            match_img = cv2.drawMatchesKnn(frame, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            note = training_set_name[best_i]
            print(f'\nDetected denomination: {note}')

            cv2.imshow('DETECTED MATCH', match_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print(f'\nNo Good Matches, closest one has {max_matches} matches ({kp_perc}%)')

            closest_match = cv2.drawMatchesKnn(frame, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            note = training_set_name[best_i]
            cv2.imshow('NO MATCH', closest_match)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print('\nProgram exited')

def preprocess(img, showImages=True):
    # Add your preprocessing steps here
    return img

if __name__ == '__main__':
    main()
