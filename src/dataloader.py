import cv2
import numpy as np
import copy as c
import math
import scipy.spatial
from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

class MOVE:
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 


    def __init__(self, video_path:str=None, frames=None):
        if isinstance(frames, list) or isinstance(frames, np.ndarray):
            self.frames = frames
        else:
            self.frames = self.load_mov_as_array(video_path)
        # self.__original = self.frames
        self.shape = self.frames[0].shape

    @classmethod
    def saveImg(cls, img, name:str):
        cv2.imwrite(f"../saved_imgs/{name}.png", img)

    def saveFrames(self, name:str):
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(f"{name}.mov", fourcc, 30, (self.frames[0].shape[1],self.frames[0].shape[0]))
        for frame in self.frames:
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()

        
    @classmethod
    def show(self, frame):
        """Shows the inputted image"""
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_mov_as_array(self, video_path):
        """Loads the frames of a .mov file into the MOVE object given its filepath/name"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
        frames = []
        while True:
            ret, frame = cap.read() # Reads a frame
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    

    def play_video(self):
        """Plays the processed MOVE recording"""

        for frame in self.frames:
            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    
    def create_average_img(self):
        """Generates the time-averaged background image of self.frames"""
        out_img = self.frames[0]
        for i in range(len(self.frames[1:])):
            out_img = cv2.addWeighted(out_img*i/(i+1), 0, self.frames[i]/(i+1), 1, 0)
        
        return out_img


    def copy(self):
        """Makes a deep copy of the object"""
        dupe = c.deepcopy(self)
        return dupe
    

    def for_all_frames(func): 
        def wrapper(self, inplace=False, *args, **kwargs):
            out_frames = [func(self, frame, *args, **kwargs) for frame in self.frames]
            if inplace:
                self.frames = out_frames
                return self
            else:
                output:MOVE = self.copy()
                output.frames = out_frames
                return output
        return wrapper
    

    # -----------------------
    # COLOR FILTERS
    # -----------------------

    @for_all_frames  
    def to_gray(self, frame):
        """Convert image to grayscale"""
        if frame.ndim == 2:
            return frame
        self.shape = self.shape[:2]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @for_all_frames
    def ranger(self, frame, minval, maxval):
        return cv2.inRange(frame, lowerb=minval, upperb=maxval)
    
    @for_all_frames 
    def color_reduce(self, frame, K=4):
        """DO NOT USE. FAR TOO COMPUTATIONALLY EXPENSIVE"""
        Z = np.float32(frame.reshape((-1, 3)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(frame.shape)

        return quantized
    
    @for_all_frames
    def hole_filler(self, frame):
        "we're doing stuff guys"

        # do flood fill
        frame = np.concatenate((np.zeros((1, frame.shape[1])), frame), axis=0)
        im_floodfill = frame.copy()
        im_floodfill = np.astype(im_floodfill, np.uint8)
        h, w = frame.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        im_floodfill_inv = cv2.bitwise_not(np.astype(im_floodfill[1:], np.uint8))
        frame = np.astype(frame[1:], np.uint8)
        return (frame | im_floodfill_inv)
    
    @for_all_frames
    def crop(self, frame, row_start = None, row_end = None, col_start = None, col_end = None):
        return frame[row_start:row_end, col_start:col_end]
        

    # ---------------------------------
    #  BLURRING FILTERS
    # ---------------------------------
    
    @for_all_frames  
    def gauss_blur(self, frame, ksize=(3,3), sigmaX=3, **kwargs):
        return cv2.GaussianBlur(frame, ksize, sigmaX, **kwargs)
    
    @for_all_frames   
    def median_blur(self, frame, ksize=3, **kwargs):
        
        return cv2.medianBlur(frame, ksize=ksize, **kwargs)
    
    @for_all_frames  
    def sharpen(self, frame, ddepth=-1, **kwargs):
        return cv2.filter2D(frame, ddepth=ddepth, kernel=self.sharpening_kernel, **kwargs)
    
    @for_all_frames
    def edge_blur(self, frame, **kwargs):
        return cv2.edgePreservingFilter(frame, **kwargs)
    

    # ---------------------------------
    #  MORPHOLOGICAL TRANSFORMATIONS
    # ---------------------------------

    @for_all_frames
    def change_contrast(self, frame, alpha=1, beta=0, **kwargs):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta, **kwargs)

    @for_all_frames  
    def dilate(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3), **kwargs):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.dilate(frame, element, **kwargs)
    
    @for_all_frames  
    def erode(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3), **kwargs):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.erode(frame, element, **kwargs)
    
    @for_all_frames
    def opening(self, frame, ksize=(3,3), **kwargs):
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, ksize, **kwargs)
    
    @for_all_frames
    def closing(self, frame, ksize=(3,3), iterations=1, **kwargs):
        # kernel = np.ones(ksize, np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=iterations, **kwargs)
    
    @for_all_frames
    def fin_connection(self, frame, ksize=(8, 8), iterations=1):
        k = np.ones(ksize, np.uint8)
        frame = cv2.dilate(frame, k, iterations = iterations)
        return cv2.erode(frame, k, iterations = iterations)

    @for_all_frames
    def fin_removal(self, frame, ksize=(8, 8), iterations=1):
        k = np.ones(ksize, np.uint8)
        frame = cv2.erode(frame, k, iterations = iterations)
        return cv2.dilate(frame, k, iterations = iterations)
    
    @for_all_frames
    def fin_retrieval(self, frame, horiz_ksize = (2, 9), vert_ksize = (12, 2), filter = False, filter_ksize = (5, 5), iterations=1):
        k1 = np.ones(horiz_ksize, np.uint8)
        k2 = np.ones(vert_ksize, np.uint8)
        k3 = np.ones(filter_ksize, np.uint8)

        frame_dupe = frame
        fin_removed = cv2.erode(frame_dupe, k1, iterations = iterations)
        fin_removed = cv2.dilate(fin_removed, k1, iterations = iterations)
        fin_removed = cv2.bitwise_not(fin_removed)

        fin = cv2.bitwise_and(fin_removed, frame)
        fin = cv2.erode(fin, k2, iterations = iterations)
        fin = cv2.dilate(fin, k2, iterations = iterations)

        if not filter:
            return fin
        else:
            fin = cv2.erode(fin, k3, iterations = iterations)
            return cv2.dilate(fin, k3, iterations = iterations)
        
    @for_all_frames
    def contouring(self, frame, thresholding = True, threshold = 200, topn = 1, checkmode=False, px_thresh = 10, topbound=10):
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blank = np.zeros(frame.shape)
        big_contours = []

        if thresholding:
            for contour in contours:
                if cv2.contourArea(contour) > threshold:
                    big_contours.append(contour)

        else:
            contour_list = []
            area_list = []

            for contour in contours:
                contour_list.append(contour)
                area_list.append(cv2.contourArea(contour))

            n_count = 0
            while n_count < topn and len(area_list) > 0:
                max_a = max(area_list)
                i = area_list.index(max_a)
                n_count += 1

                if max_a > 100:
                    big_contours.append(contour_list[i])
                area_list.pop(i)
                contour_list.pop(i)
        
        newframe = cv2.drawContours(blank, big_contours, -1, (255, 255, 255), 1)

        if not checkmode:
            return newframe
        else:
            row = newframe.shape[0] - 1
            leftmost = -1
            rightmost = -1
            while row >= 0:
                values, counts = np.unique(newframe[row], return_counts=True)
                cols = []
                if row <= topbound:
                    newframe[row] = np.zeros_like(newframe[row])
                elif len(counts) > 1 and counts[-1] > 2:
                    for col in range(newframe.shape[1]):
                        if newframe[row][col] == 255:
                            cols.append(col)
                    dists = []
                    for j in range(len(cols)):
                        col = cols[j]
                        notfound = True
                        within = True
                        off = 1
                        while notfound and off < 30:
                            r = row + 1
                            c = max(col - off, 0)
                            while notfound and r < (row + off + 1) and r < newframe.shape[0]:
                                while notfound and c < (col + off + 1) and c < newframe.shape[1]:
                                    if newframe[r][c] == 255:
                                        notfound = False
                                        dists.append(off)
                                    c += 1
                                r += 1
                            off += 1
                        if notfound:
                            dists.append(31)
                    while len(cols) > 2:
                        i = dists.index(max(dists))
                        specialcol = cols[i]
                        newframe[row][specialcol] = 0
                        cols.pop(i)
                        dists.pop(i) 
                elif len(counts) > 1 and counts[-1] > 0:
                    for col in range(newframe.shape[1]):
                        if newframe[row][col] == 255:
                            cols.append(col)
                if len(cols) > 0:
                    if leftmost < 0:
                        leftmost = min(cols)
                        rightmost = max(cols)
                    else:
                        if len(cols) > 1:
                            if abs(leftmost - min(cols)) > px_thresh:
                                newframe[row][min(cols)] = 0
                            else:
                                leftmost = min(cols)
                                if abs(rightmost - leftmost) > (2*px_thresh):
                                    rightmost = leftmost + px_thresh 
                            if abs(rightmost - max(cols)) > px_thresh:
                                newframe[row][max(cols)] = 0
                            else:
                                rightmost = max(cols)
                                if abs(rightmost - leftmost) > (2*px_thresh):
                                    leftmost = rightmost - px_thresh 
                        else:
                            if abs(leftmost - cols[0]) > px_thresh and abs(rightmost - cols[0]) > px_thresh:
                                newframe[row][cols[0]] = 0    
                            else:
                                leftmost = cols[0]
                                rightmost = cols[0]

                row -= 1
            return newframe
        
    def fancy_contouring(self, topn=3, threshold=5, comparen=3):
        '''thought this would make things better. made them worse and slower. do not recommend'''
        saved_frames = []

        for i in range(len(self.frames)):
            frame = self.frames[i]
            contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contour_list = []
            area_list = []

            for contour in contours:
                contour_list.append(contour)
                area_list.append(cv2.contourArea(contour))

            big_contours = []
            n_count = 0
            while n_count < topn and len(area_list) > 0:
                max_a = max(area_list)
                q = area_list.index(max_a)
                n_count += 1

                if max_a > 100:
                    big_contours.append(contour_list[q])
                area_list.pop(q)
                contour_list.pop(q)
            
            temp_frame = np.zeros_like(frame)
            cv2.drawContours(temp_frame, big_contours, -1, 255, 1)

            if i > (comparen - 1):
                for row in range(temp_frame.shape[0]):
                    if np.isin(255, temp_frame[row]):
                        for col in range(temp_frame.shape[1]):
                            if temp_frame[row][col] == 255:
                                in_thresh = False
                                for n in range(-1*comparen, 0):
                                    j = max(row - threshold, 0)
                                    k = max(col - threshold, 0)
                                    while j < row + threshold and j < temp_frame.shape[0] and in_thresh == False:
                                        if np.isin(255, saved_frames[-1][j][(col - threshold):(col + threshold)]):
                                            while k < col + threshold and k < temp_frame.shape[1] and in_thresh == False:
                                                if saved_frames[-1][j][k] == 255:
                                                    in_thresh = True
                                                k += 1
                                        j += 1
                                if not in_thresh:
                                    temp_frame[row][col] = 0
            
            if not np.isin(255, temp_frame):
                print(i)
                cv2.drawContours(temp_frame, big_contours, -1, 255, 1)
            
            saved_frames.append(temp_frame)
        
        return saved_frames
        

    def linfit(self, x, a, b):
        return a*x + b
    
    def quartfit(self, x, a, b, c, d, e):
        return a*np.power(x, 4) + b*np.power(x, 3) + c*np.power(x,2) + d*np.power(x, 1) + e


    def frame_to_xy(self, frame):
        y = []
        x = []
        half_width = []

        for row in range(0, frame.shape[0]):
            active_row = frame[row]
            if np.isin(255, active_row):
                leftmost = -1
                rightmost = -1
                col = 0
                while leftmost < 0 and col < frame.shape[1]:
                    if active_row[col] == 255:
                        leftmost = col
                    col += 1
                col = frame.shape[1] - 1
                while rightmost < 0 and col >= 0:
                    if active_row[col] == 255:
                        rightmost = col
                    col -= 1
                x.append(float(row))
                center =  (leftmost + rightmost) / 2.0
                y.append(float(center))
                half_width.append((rightmost - leftmost) / 2.0)
    
        return x, y, half_width
        

    def lin_tracking(self, sig=2, r=2):

        lowest_point = -1
        p0 = []
        a = []
        b = []
        fail_count = 0


        for i in range(0, len(self.frames)):
            active_frame = self.frames[i]

            x, y, half_widths = self.frame_to_xy(active_frame)

            if i == 0:
                x = np.array(x)
                yerr = np.array(y) / 10
                lowest_point = max(x)

                (popt, pcov) = curve_fit(self.linfit, x, y, sigma=yerr, absolute_sigma=True)
                p0 = popt
                a.append(popt[0])
                b.append(popt[1])  

            else:
                x = np.array(x)
                yerr = 0.1 * np.array(y)

                try:
                    (popt, pcov) = curve_fit(self.linfit, x, y, sigma=yerr, absolute_sigma=True)
                    p0 = popt
                except:
                    popt = p0
                    fail_count += 1
                    print(i)
                a.append(p0[0])
                b.append(p0[1])
        
        a = gaussian_filter1d(a, sigma=sig, radius=r)
        b = gaussian_filter1d(b, sigma=sig, radius=r)
        
        return a, b, lowest_point
    

    def create_lin_mask(self, mask, a, b, lowest_point, offset:int, display=False, display_vid=None, display_color=(232, 235, 52)):
        output_frames = []
        display_frames = []
        lowest_point = int(lowest_point)

        for frame in range(len(self.frames)):
            output_frame = 255 * np.ones_like(self.frames[frame])
            for row in range(lowest_point + 10):
                center = int(a[frame] * row + b[frame])
                output_frame[row][(center - offset):(center + offset)] = mask[frame][row][(center - offset):(center + offset)]
            if display:
                if display_vid != None:
                    display_frame = display_vid.frames[frame].copy()
                else:
                    display_frame = self.frames[frame].copy()
                top_point = int(b[frame])
                bottom_point = int(a[frame] * (lowest_point + 10) + b[frame])
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                cv2.line(display_frame, (top_point - offset, 0), (bottom_point - offset, lowest_point + 10), display_color, 1)
                cv2.line(display_frame, (top_point + offset, 0), (bottom_point + offset, lowest_point + 10), display_color, 1)
                display_frames.append(display_frame)
            output_frames.append(output_frame)

        if display:
            return output_frames, display_frames
        else:
            return output_frames
        
    def quart_tracking(self, failmode, seed=False, b=[], asymp_thresh = 10):
        lowest_point = -1
        p0 = []
        params = np.zeros((5, len(self.frames)))

        for i in range(0, len(self.frames)):
            active_frame = self.frames[i]
            x, y, half_widths = self.frame_to_xy(active_frame)
            save_point = (0, 0)
            if len(x) > 40:
                x = np.array(x)
                y = np.array(y)
                if not np.isin(0, x) and i < len(b):
                    x = np.concatenate([[0], x])
                    y = np.concatenate([[b[i]], y])


                if i == 0:
                    lowest_point = max(x)
                    (popt, pcov) = curve_fit(self.quartfit, x, y)
                    p0 = popt
                    for j in range(5):
                        params[j][i] = p0[j] 

                else:
                    try:
                        if seed:
                            (popt, pcov) = curve_fit(self.quartfit, x, y, p0=p0)
                        else:
                            (popt, pcov) = curve_fit(self.quartfit, x, y, sigma=(5*np.ones_like(y)), absolute_sigma=True)

                        if abs(self.quartfit(0, *popt) - self.quartfit(1, *popt)) > asymp_thresh:
                            print(i)
                        else:
                            p0 = popt
                    except:
                        print(i)
                    for j in range(5):
                        params[j][i] = p0[j]
                
            else:
                for j in range(5):
                    params[j][i] = p0[j]
        
        return params, lowest_point
    
    def create_quad_mask(self, mask, params, fitfunc, lowest_point, offset:int, display=False, display_vid=None, display_color=(232, 235, 52)):
        output_frames = []
        display_frames = []
        lowest_point = int(lowest_point)

        for frame in range(len(self.frames)):
            params_by_frame = [params[0][frame], params[1][frame], params[2][frame], params[3][frame], params[4][frame]]
            output_frame = 255 * np.ones_like(self.frames[frame])
            for row in range(lowest_point + 10):
                center = int(fitfunc(row, *params_by_frame))
                output_frame[row][(center - offset):(center + offset)] = mask[frame][row][(center - offset):(center + offset)]
            if display:
                if display_vid != None:
                    display_frame = display_vid.frames[frame].copy()
                else:
                    display_frame = self.frames[frame].copy()
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                for row in range(lowest_point + 10):
                    center = int(fitfunc(row, *params_by_frame))
                    if (center - offset) >= 0 and (center - offset) < display_frame.shape[1]:
                        display_frame[row][center - offset] = display_color
                    if (center + offset) >= 0 and (center + offset) < display_frame.shape[1]:
                        display_frame[row][center + offset] = display_color
                display_frames.append(display_frame)

            output_frames.append(output_frame)

        if display:
            return output_frames, display_frames
        else:
            return output_frames



    # ---------------------------------
    #  FEATURE DETECTION
    # ---------------------------------

    @for_all_frames  
    def edges(self, frame, thresh1=25, thresh2=100, **kwargs):
        return cv2.Canny(frame, thresh1, thresh2, **kwargs)
    
    @for_all_frames  
    def connect_le_components(self, frame, connectivity=8, ltype=cv2.CV_32S, **kwargs):
        _, out = cv2.connectedComponents(frame, connectivity=connectivity, ltype=ltype, **kwargs)
        return out
    
    @for_all_frames
    def img_like(self, frame, img):
        """Replaces each frame with img"""
        return img
    
    def isin(self, a, b):
        if a[0] <= b[0] and a[1] <= b[1]:
            if (a[0] + a[2]) >= (b[0] + b[2]):
                if (a[1] + a[3]) >= (b[1] + b[3]):
                    return True
        return False
    
    def remove_nested(self, msers, bboxes):
        i = 0
        mser_list = []
        for mser in msers:
            mser_list.append(mser)

        while i < len(bboxes):
            j = 0
            to_delete = []
            decrement = 0

            while j < len(bboxes) and i < len(bboxes):
                if i != j and self.isin(bboxes[i], bboxes[j]):
                    to_delete.append(j)
                    if j < i:
                        decrement += 1
                j += 1
    
            bboxes = np.delete(bboxes, to_delete, axis=0)
            for k in range(len(to_delete)):
                index = to_delete[len(to_delete) - k - 1]
                mser_list.pop(index)
            i += (1 - decrement)
        return (mser_list, bboxes)
    
    def mser_to_array(self, mser, frame_like):
        frame = np.zeros_like(frame_like)
        for point in mser:
            col, row = point
            frame[row][col] = 255
        return frame
    
    def array_to_mser(self, mser_array):
        mser = []
        for row in range(mser_array.shape[0]):
            if np.isin(255, mser_array[row]):
                for col in range(mser_array.shape[1]):
                    if mser_array[row][col] == 255:
                        mser.append([col, row])
        return mser

    def remove_waves(self, msers, bboxes, frame_like, wave_thresh):
        i = 0
        hthresh = wave_thresh[0]
        vthresh = wave_thresh[1]

        mser_list = []
        for mser in msers:
            mser_list.append(mser)

        while i < len(bboxes):
            x, y, w, h = bboxes[i]
            if (x + w/2) < hthresh or (x + w/2) > (frame_like.shape[1] - hthresh):
                if (x + w) < hthresh or x > (frame_like.shape[1] - hthresh):
                    bboxes = np.delete(bboxes, i, 0)
                    msers.pop(i)
                    i -= 1
                else:
                    mser_array = self.mser_to_array(msers[i], frame_like)
                    cols = list(range(0, hthresh)) + list(range(frame_like.shape[1] - hthresh, frame_like.shape[1]))
                    for col in cols:
                        row = y + h
                        found = False
                        while not found and row >= 0:
                            if row < vthresh:
                                mser_array[row][col] = 0
                            elif mser_array[row][col] == 255:
                                found = True
                            row -= 1
                    mser_list[i] = self.array_to_mser(mser_array)
                    xlist = []
                    ylist = []
                    for point in mser_list[i]:
                        xlist.append(point[0])
                        ylist.append(point[1])
                    if len(xlist) > 200 and len(ylist) > 200:
                        bboxes[i] = [min(xlist), min(ylist), max(xlist) - min(xlist), max(ylist) - min(ylist)]
                    else:
                        bboxes = np.delete(bboxes, i, 0)
                        mser_list.pop(i)
                        i -= 1
            i += 1
        return (mser_list, bboxes)


    def mser_track(self, mser, display=True, inplace=False, vert_threshold=0, area_threshold=0.5):
        prev_area = 0
        
        if not inplace:
            vid = self.copy()
        else:
            vid = self

        for frame in vid.frames:
            for i in range(0, 25):
                scale = 1.75 * math.cos((3.1/50) * i) + 1
                frame[i] = cv2.convertScaleAbs(frame[i], alpha=scale, beta=0).reshape(1264,)
            frame = cv2.convertScaleAbs(frame, alpha = 1.5, beta = 20)
    
        display_vid = []
        bboxes_by_frame = []
        msers_by_frame = []

        for frame in vid.frames: 
            vid_frame = frame.copy()
            (msers, bboxes) = mser.detectRegions(frame)
            if len(msers) > 0:
                (msers, bboxes) = self.remove_nested(msers, bboxes)
                # comparison

                #if len(prev_box) > 0:
                    #frame_params = []
                    #for bbox in bboxes:
                    #    x0, y0, w0, h0 = bbox
                    #    frame_params.append([w0, h0, w0 * h0, x0 + (w0/2), y0 + (h0/2)])
                    #prev_params = [w, h, w * h, x + (w/2), y + (h/2)]

                    #dists = []
                    #for i in range(len(frame_params)):
                    #    dist0 = distance.euclidean(frame_params[i], prev_params)
                    #    dists.append(dist0)
                    #prev_box = bboxes[dists.index(min(dists))]
                    #x, y, w, h = prev_box
                
                area_list = []
                bboxes_kept = []
                msers_kept = []
                cutoff = vert_threshold * frame.shape[0]
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    x, y, w, h = bbox
                    area = w * h
                    area_list.append(area)
                    if area > (area_threshold * prev_area):
                        if vert_threshold > 0 and (y + h/2) < cutoff:
                            bboxes_kept.append(bbox)
                            msers_kept.append(msers[i])
                        elif vert_threshold == 0:
                            bboxes_kept.append(bbox)
                            msers_kept.append(msers[i])
                bboxes_by_frame.append(bboxes_kept)
                msers_by_frame.append(msers_kept)

                if display:
                    vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
                    for bbox in bboxes_kept:
                        x, y, w, h = bbox
                        cv2.rectangle(vid_frame, (x, y), (x + w, y + h), (244, 255, 41), 1)
                    display_vid.append(vid_frame)

                if len(area_list) > 0:
                    prev_area = max(area_list)
                else:
                    prev_area = 0
            else:
                bboxes_by_frame.append([])
                msers_by_frame.append([])
        
        return (display_vid, bboxes_by_frame, msers_by_frame)
    

    # ---------------------------------
    #  Masking Utilities
    # ---------------------------------

    def get_avg(self, skipframes=0) -> np.ndarray:
        """
        Creates the time-average image to act as a pseudo-background 
        (static) image

        :param skipframes: Optional parameter to dictate how many frames 
            to skip from the start to prevent over-representation of the
            starting fin position in the pseudo-background image
        """
        return np.mean(self.frames[skipframes:], axis=0).astype('uint8')
    
    
    def remove_avg(self, skipframes=0, happy_accident_mode=True):
        """
        Subtracts the average image from all frames

        :param skipframes: Optional parameter to dictate how many frames 
            to skip from the start to prevent over-representation of the
            starting fin position in the pseudo-background image
        :param happy_accident_mode: When False, this mode takes the absolute
            difference between the two images it compares. When enabled, the
            function merely subtracts the average image from a given image,
            meaning that when the average image is brighter than the given
            image, the pixel's value becomes negative which wraps around to
            positive, making these spots extremely bright. This had an
            unintentionally positive effect that greatly improved the 
            pipeline's results
        """

        img_avg = self.get_avg(skipframes=skipframes)
        
        updated_frames = []
        for frame in self.frames:
            if happy_accident_mode:
                frame = np.where(frame <= img_avg, img_avg - frame, np.zeros_like(frame))
            else:
                frame = cv2.absdiff(frame, img_avg)

            updated_frames.append(frame.astype('uint8'))

        self.frames = updated_frames


    def apply_mask(self, other, inplace=False):
        """
        Overlays a mask video that highlights the masked regions of each frame
            and darkens unmasked regions
        
        :param other: The 
        :param inplace: Description
        """
        out = []
        for frame in zip(self.frames, other.frames):
            framey = frame[0]
            mask = frame[1]
            mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            framey = cv2.addWeighted(cv2.bitwise_and(framey, mask_3_channel), 2, framey, 0.5, 1)
            out.append(framey)
        
        if inplace:
            self.frames = out
        return out
    

    def combine_masks(self, other, inplace=False):
        """Combines two masks"""
        out = []
        for frame in zip(self.frames, other.frames):
            frameS, frameO = frame
            maskS = np.bitwise_and(frameS, np.ones_like(frameS))
            maskO = np.bitwise_and(frameO, np.ones_like(frameS))
            multi_mask = (127*np.where(maskS==1, 2*maskS, maskO)).astype('uint8')
            out.append(multi_mask)

        if inplace:
            self.frames = out
        return out
    
    def bitwise_and(self, other, inplace=False):
        out = []
        for frame in zip(self.frames, other.frames):
            vid, mask = frame
            out.append(cv2.bitwise_and(vid, mask))
        return out
    
    @for_all_frames
    def generic_filter(self, frame, /, func, **kwargs):
        return func(frame, **kwargs)

    def __getitem__(self, index):
        img = self.frames[index]
        return img
    

    def __eq__(self, other) -> bool:
        if not isinstance(other, MOVE):
            return TypeError
        if self.shape != other.shape:
            return False
        
        for self_frame, other_frame in zip(self.frames, other.frames):
            if self_frame.shape != other_frame.shape:
                return False
            if not (np.sum(self_frame==other_frame) == self_frame.size):
                return False
        return True
        

MOVE.median_blur.__doc__ = cv2.medianBlur.__doc__
MOVE.sharpen.__doc__ = cv2.filter2D.__doc__
MOVE.edge_blur.__doc__ = cv2.edgePreservingFilter.__doc__
MOVE.change_contrast.__doc__ = cv2.convertScaleAbs.__doc__
MOVE.edges.__doc__ = cv2.Canny.__doc__
MOVE.connect_le_components.__doc__ = cv2.connectedComponents.__doc__


    