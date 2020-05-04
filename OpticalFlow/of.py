import numpy as np
import cv2 as cv
import csv
import os.path
from os import path
# Used to calculate time taken to complete script
from datetime import datetime

# Parameters for lucas kanade optical flow (based from opencv example)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create circle and line colours
colorC = (255, 0, 0)
colorL = (0, 255, 0)
colorB = (0, 0, 255)

class OpticalFlow:
    def __init__(self, video_src):
        self.cap = cv.VideoCapture(video_src)

    def run(self):
        # Take first frame and convert to gray
        ret, old_frame = self.cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # Create specific optical flow points
        process_width = old_frame.shape[1]
        process_height = old_frame.shape[0]
        cell_height = old_frame.shape[0]/5
        cell_width = old_frame.shape[1]/10

        inter_block_dist_x = (process_width - cell_width)/9
        inter_block_dist_y = (process_height - cell_height)/4
        offset_x = (process_width - (inter_block_dist_x*9))/2
        offset_y = (process_height - (inter_block_dist_y*4))/2

        tp_vec = []

        for j in range(0, 5):
            for i in range(0, 10):
                cx = i*inter_block_dist_x + offset_x
                cy = j*inter_block_dist_y + offset_y
                tp_vec.append([[cx, cy]])

        p0 = np.asarray(tp_vec, dtype=np.float32)
        disp_avg = np.asarray(tp_vec, dtype=np.float32)

        with open('opticalflow_result.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            #150 columns per frame
            frame_num = 0
            while True:
                row = []
                ret, frame = self.cap.read()
                if ret:
                    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    mask = frame.copy()
                    
                    # Display Current Frame Number
                    font = cv.FONT_HERSHEY_SIMPLEX 
                    org = (50, 50) 
                    fontScale = 1
                    thickness = 2
                    mask = cv.putText(mask, str(frame_num), org, font, fontScale, colorC, thickness, cv.LINE_AA) 

                    for i in p0:
                        cv.circle(mask, (int(i[0][0]), int(i[0][1])), 4, colorB, -1)

                    # Calculate optical flow
                    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    all_new = p1
                    all_old = p0

                    good_indexes = np.argwhere(st.flatten() == 1)

                    disp_avg[good_indexes] = (0.99 * disp_avg[good_indexes]) + (0.01 * p1[good_indexes])

                    # Draw the tracks
                    for i, (new, old) in enumerate(zip(disp_avg, all_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        status = st.item(i)
                        #if (status == 1):
                        cv.circle(mask, (int(c), int(d)), 4, colorC, -1)
                        cv.line(mask, (int(c), int(d)), (int(a), int(b)), colorL, 2)
                        row.extend([status, c-a, d-b])

                    # Now update the previous frame
                    writer.writerow(row)
                    old_gray = frame_gray.copy()
                    cv.imshow('Video Frame', mask)
                    frame_num += 1
                    # Exit if escape key pressed
                    ch = cv.waitKey(1)
                    if ch == 27:
                        break
                else:
                    cv.destroyAllWindows()
                    self.cap.release()
                    break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        print('---| Please enter the video file you wish to perform optical flow calculations on as an argument to this script! |---')
        print('---| For example `python of.py "video_file.mp4"` |---')
        sys.exit()

    if path.exists(video_src):
        startTime = datetime.now()
        OpticalFlow(video_src).run()
        print('---| Optical Flow Completed |---')
        print("Time to complete: " + str(datetime.now() - startTime))
    else:
        print('---| File ' + video_src + ' not found! |---')

if __name__ == '__main__':
    main()