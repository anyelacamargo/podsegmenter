# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:36:28 2020

@author: X992029
"""

#!/usr/bin/python

import os
import sys
import cv2
import math
import numpy as np
import utils
from matplotlib import pyplot as plt
import glob
import imutils
from numpy import linalg



class Segmenter(object):
 
    
    def __init__(self) :
        '''
        '''
    
    
    def init2(self, image_dir, key_frame, output_dir, img_filter=None):
        '''
        image_dir: 'directory' containing all images
        key_frame: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        optional: 
           img_filter = 'JPG'; None->Take all images
        '''
    
        self.key_frame_file = os.path.split(key_frame)[-1]
        self.output_dir = output_dir
        
       
        # Open the directory given in the arguments
        self.dir_list = []
        try:
            self.dir_list = os.listdir(image_dir)
            if img_filter:
                # remove all files that doen't end with .[image_filter]
                self.dir_list = filter(lambda x: x.find(img_filter) > -1, self.dir_list)
            try: #remove Thumbs.db, is existent (windows only)
                self.dir_list.remove('.DS_Store')
            except ValueError:
                pass
                
        
        except:
            print >> sys.stderr, ("Unable to open directory: %s" % image_dir)
            sys.exit(-1)
    
        self.dir_list = map(lambda x: os.path.join(image_dir, x), self.dir_list)
        #print(list(self.dir_list))
        self.dir_list = list(filter(lambda x: x != key_frame, self.dir_list))
        
        base_img_rgb = cv2.imread(key_frame)
        
        
        if (base_img_rgb == None).any():
            raise IOError("%s doesn't exist" %key_frame)
        
        final_img = self.stitch(base_img_rgb, 0)        
        


    def filter_matches(self, matches, ratio = 0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        
        return filtered_matches
    
    def imageDistance(self, matches):
    
        sumDistance = 0.0
    
        for match in matches:
    
            sumDistance += match.distance
    
        return sumDistance
    
    def plot_image(img):
        """
        Parameters
        ----------
        img : image (RGB)
            image
               
        """
        plt.figure()
        plt.axis("on")
        plt.imshow(img)
        plt.show()
        
    
    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)
    
        (y, x) = image.shape[:2]
    
        base_p1[:2] = [0,0]
        base_p2[:2] = [x,0]
        base_p3[:2] = [0,y]
        base_p4[:2] = [x,y]
    
        max_x = None
        max_y = None
        min_x = None
        min_y = None
    
        for pt in [base_p1, base_p2, base_p3, base_p4]:
    
            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
    
            hp_arr = np.array(hp, np.float32)
    
            normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
    
            if ( max_x == None or normal_pt[0,0] > max_x ):
                max_x = normal_pt[0,0]
    
            if ( max_y == None or normal_pt[1,0] > max_y ):
                max_y = normal_pt[1,0]
    
            if ( min_x == None or normal_pt[0,0] < min_x ):
                min_x = normal_pt[0,0]
    
            if ( min_y == None or normal_pt[1,0] < min_y ):
                min_y = normal_pt[1,0]
    
        min_x = min(0, min_x)
        min_y = min(0, min_y)
    
        return (min_x, min_y, max_x, max_y)

    
    def stitch(self, base_img_rgb, round=0):
    
        plot_image(base_img_rgb)
        print(round)
        #print(self.dir_list)
        if ( len(self.dir_list) < 1 ):
            return base_img_rgb 
    
    
        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)
    
        # Use the SIFT feature detector
        detector = cv2.xfeatures2d.SIFT_create()
    
        # Find key points in base image for motion estimation
        base_features, base_descs = detector.detectAndCompute(base_img, None)
    
        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
            trees = 5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
    
        print("Iterating through next images...")
    
        closestImage = None
    
        next_img_path = self.dir_list[0]
        print(next_img_path)
        #sprint("Reading %s..." % next_img_path)

        # Read in the next image...
        next_img_rgb = cv2.imread(next_img_path)
        next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

        print("\t Finding points...")

        # Find points in the next frame
        next_features, next_descs = detector.detectAndCompute(next_img, None)

        matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

        #sprint("\t Match Count: ", len(matches))

        matches_subset = self.filter_matches(matches)

        #print "\t Filtered Match Count: ", len(matches_subset)

        distance = self.imageDistance(matches_subset)

        #print "\t Distance from Key Image: ", distance

        averagePointDistance = distance/float(len(matches_subset))

        #print "\t Average Distance: ", averagePointDistance

        kp1 = []
        kp2 = []

        for match in matches_subset:
            kp1.append(base_features[match.trainIdx])
            kp2.append(next_features[match.queryIdx])

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        #print '%d / %d  inliers/matched' % (np.sum(status), len(status))

        inlierRatio = float(np.sum(status)) / float(len(status))

        if ( closestImage == None or inlierRatio > closestImage['inliers'] ):
            closestImage = {}
            closestImage['h'] = H
            closestImage['inliers'] = inlierRatio
            closestImage['dist'] = averagePointDistance
            closestImage['path'] = next_img_path
            closestImage['rgb'] = next_img_rgb
            closestImage['img'] = next_img
            closestImage['feat'] = next_features
            closestImage['desc'] = next_descs
            closestImage['match'] = matches_subset
    
        #print "Closest Image: ", closestImage['path']
        #print "Closest Image Ratio: ", closestImage['inliers']

        self.dir_list = list(filter(lambda x: x != closestImage['path'], self.dir_list))
    
        H = closestImage['h']
        H = H / H[2,2]
        H_inv = linalg.inv(H)
        print(closestImage['inliers'])
    
        if ( closestImage['inliers'] > 0.01 ): # and 
    
            (min_x, min_y, max_x, max_y) = self.findDimensions(closestImage['img'], H_inv)
    
            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])
    
            move_h = np.matrix(np.identity(3), np.float32)
    
            if ( min_x < 0 ):
                move_h[0,2] += -min_x
                max_x += -min_x
    
            if ( min_y < 0 ):
                move_h[1,2] += -min_y
                max_y += -min_y
    
            #print "Homography: \n", H
            #print "Inverse Homography: \n", H_inv
            #print "Min Points: ", (min_x, min_y)
    
            mod_inv_h = move_h * H_inv
    
            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))
    
            #print "New Dimensions: ", (img_w, img_h)
    
            # crop edges
            #print "Cropping..."
            base_h, base_w, base_d = base_img_rgb.shape
            next_h, next_w, next_d = closestImage['rgb'].shape
    
            base_img_rgb = base_img_rgb[5:(base_h-5),5:(base_w-5)]
            closestImage['rgb'] = closestImage['rgb'][5:(next_h-5),5:(next_w-5)]
    
    
            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
            #print "Warped base image"
    
            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=1000, save=True, title="base_img_warp")
            # cv2.destroyAllWindows()
    
            next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))
            #print "Warped next image"
    
            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=1000, save=True, title="next_img_warp")
            # cv2.destroyAllWindows()
    
            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    
            #print "Enlarged Image Shape: ", enlarged_base_img.shape
            #print "Base Image Shape: ", base_img_rgb.shape
            #print "Base Image Warp Shape: ", base_img_warp.shape
    
            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp
    
            # Create masked composite
            (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
                0, 255, cv2.THRESH_BINARY)
    
            # add base image
            enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
                mask=np.bitwise_not(data_map), 
                dtype=cv2.CV_8U)
    
            # add next image
            final_img = cv2.add(enlarged_base_img, next_img_warp, 
                dtype=cv2.CV_8U)
    
            # Crop black edge
            final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
            dino, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #print "Found %d contours..." % (len(contours))
    
            max_area = 0
            best_rect = (0,0,0,0)
    
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
    
                deltaHeight = h-y
                deltaWidth = w-x
    
                area = deltaHeight * deltaWidth
    
                if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                    max_area = area
                    best_rect = (x,y,w,h)
    
            if ( max_area > 0 ):
                final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                        best_rect[0]:best_rect[0]+best_rect[2]]
    
                final_img = final_img_crop
    
            # output
            final_filename = "%s/%d.JPG" % (self.output_dir, round)
            cv2.imwrite(final_filename, final_img)
    
            return self.stitch(final_img, round+1)
        
        else:
            #break
            
            print('finish')
            #return self.stitch(base_img_rgb, round+1)
    
    def kmeans(self, image, segments, ch_num):
        #Preprocessing step
        #segments = 5
        image=cv2.GaussianBlur(image,(7,7),0)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vectorized=image.reshape(-1,ch_num)
        vectorized=np.float32(vectorized)
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(vectorized, segments, None, 
                                    criteria,10, cv2.KMEANS_RANDOM_CENTERS)
        if(ch_num > 1) :
            res = center[label.flatten()]
        else :
             res = center[label]
        segmented_image = res.reshape((image.shape))
        
        
        return (label.reshape((image.shape[0],image.shape[1])), 
                segmented_image.astype(np.uint8), center,vectorized)


    def plot_centroids(self, vectorized, center) :

        y = {}
        for i in range(0, center.shape[1]) :
            y[i] = vectorized[label.ravel() == i]  
        # Plot the data
            plt.scatter(y[i][:,0], y[i][:,1])
            #plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
            plt.scatter(center[:,0], center[:,1], s = 80,c = 'y', marker = 's')
    
        plt.xlabel('Height'),plt.ylabel('Weight')
        plt.show()


    def extractComponent(self, image,label_image,label):
    
        component = np.zeros(image.shape,np.uint8)
        #component[label_image==label]=image[label_image==label]
        component[label_image==label] = 1
        #print(component)
        return component


    def plot_image(self, img):
        """
        Parameters
        ----------
        img : image (RGB)
            image
               
        """
        plt.figure()
        plt.axis("on")
        plt.imshow(img)
        plt.show()
    
    
    
    
    def get_features(self, cnt):
        """
        Parameters
        ----------
       
        Returns
        -------
            
        """
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        k = cv2.isContourConvex(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
         
        x = str(str(area) + ',' + str(radius) + ',' + str(k))
        
        return(x)
        
        
        

    def postproc_custom1(self, binay_img, bx):
        """
        Parameters
        ----------
        img : image (RGB)
            image
               
        Returns
        -------
        out
            image 2D
        """
        kernel = np.ones((bx, bx),np.uint8)
        erosion = cv2.erode(binay_img, kernel, iterations = 1)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
        im_floodfill = closing.copy()
        # Mask used to flood filling.
        h, w = closing.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        #im_out = closing | im_floodfill_inv
        return(im_floodfill_inv)



    def count_clustersize(self, label):
        """
         Parameters
        ----------
        label : numpy array
               
        Returns
        -------
        K  with minimum number of pixels
            
        """
        y = list()
        for i in np.unique(label) :
            y.append(np.count_nonzero(label == i))
     
        return(y.index(min(y)))


    def plot_clusters(self, image_r, label) :
        # Show centroids
        #S.plot_centroids(vectorized, center)
        for i in range(0,k) :
            extracted = S.extractComponent(image_raw, label,i)
        #    #extracted = extracted[:,:,1]
            S.plot_image(extracted)
            #plt.imshow(extracted, cmap=plt.cm.gray)


    def filter_frame(sel, img_copy, min, max):
        """
        Parameters
        ----------
        img : image (RGB)
            image
               
        Returns
        -------
        out
            image 2D
        """
        #gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(img_copy, min, max, cv2.THRESH_BINARY)
        
        return(thresholded)
   
     
    def impixel(self, img):
        """
         Parameters
        ----------
        img : image (RGB)
            image
               
        Returns
        -------
        out
            image 2D
        """
        
        #scale_width = 640 / img.shape[1]
        #scale_height = 480 / img.shape[0]
        #scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * 1)
        window_height = int(img.shape[0] * 1)
        #
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', window_width, window_height)
        cv2.imshow('image', img)
        cv2.setMouseCallback("image", S.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows


    #right-click event value 
    def click_event(self, event, x, y, flags, param):
       
        if event == cv2.EVENT_LBUTTONDOWN:
            a = img.shape
            d = list()
            #print(a[2])
            for i in range(1, a[1]):
                d.append(img[y,x])
            
            red = img[y,x]
                     
            print(red)
 
       
    def show_segments(self, img, cnts) :
        """
         Parameters
        ----------
        img : image (RGB) image
        cnts : segments extracted from processed image
               
        Returns
        -------
        
        """
        window_width = int(img.shape[1] * 0.1)
        window_height = int(img.shape[0] * 0.1)
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        for i, c in enumerate(cnts):
             	# draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cv2.drawContours(image_raw1, [c], -1, (0, 255, 0), 2)
            
        cv2.resizeWindow('Image', window_width, window_height)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows

        
    def extract_segments(self, img) :   
        """
        Extract countourns from segmented images
        Parameters
        ----------
        img : image (RGB) image
        
               
        Returns
        -------
        cnts : segments extracted from processed image
        
        """
        cnts = cv2.findContours(extracted, cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        print("[INFO] {} unique contours found".format(len(cnts)))
        return(cnts)
        
        
# Path root 
root_path = 'c:/anyela/repo/podsegmenter/'        
# Path output folder
output_dir = 'c:/anyela/repo/podsegmenter/output/'
# Image locations
folder_names = ['test']



S = Segmenter()
# The best cluster size to segment target images
k = 7


# Create csv file
f = open(output_dir + 'dummy' + '.csv', 'w')
x = str('folder_name' + ',' + 'image_name' +',' + 'area'  +',' +  'radius'  +',' +  
        'convex') 
f.write(x + '\n')
# Loop through folders
for fname in folder_names :
    file_list = os.path.abspath(root_path + fname + '/' + '*.JPG')
    images = sorted(glob.glob(file_list))
    # Loop through images
    for iname in images[0:1]:
        print(iname)
        image_raw1 = cv2.imread(iname)
        height, width = image_raw1.shape[:2]
        image_raw1 = cv2.resize(image_raw1, (round(width / 2), round(height / 2)))
        image_raw = image_raw1[:,:,0]
       
        label,result, center, vectorized = S.kmeans(image_raw, k, 1)
        m = S.count_clustersize(label)
        
        extracted = S.extractComponent(image_raw, label, m)
        #extracted = S.postproc_custom1(extracted, 5)
        nb_comp,output,sizes,centroids = cv2.connectedComponentsWithStats(extracted, 
                                                                     connectivity=4)
        
        #print("[INFO] {} unique components found".format(centroids))
        # find contours in the thresholded image
        cnts = S.extract_segments(extracted)
        for cnt in cnts :
            x = S.get_features(cnt)
            x = fname + ',' + iname.split("\\")[-1] + ',' + x
            f.write(x + '\n')
        #S.show_segments(image_raw1, cnts)

f.close()
        
       



