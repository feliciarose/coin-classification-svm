'''
Computer Vision Project 3: Felicia Drysdale U43237407
In this project, I applied machine learning techniques to classify different types of coins by processing images. 
I manually reviewed and prepared the test images by erasing the background in each image and replacing it with different backgrounds,
leaving only the coin visible. I needed more data to improve accuracy, so I wrote a script that randomly
augmented the data that I had and saved it as a new image, thus I had more data to train the SVM model. The augmentation
involved a random selection of blurring, flipping, adding noise, lightening and darkening the images. After I 
created more data, I used Canny edge detection, to extract important features such as radius and area (thousandths decimal) 
from the contours of the coins. These features were then used to train a Support Vector Machine (SVM) classifier to differentiate 
between different coin types: quarters, pennies, dimes, and nickels. The images were resized, scaled, and blurred to improve the model's performance. 
By manually preparing the dataset and focusing on the key features of each coin, I was able to train the model, achieving an accuracy of 93.02%.
Note: I ended up having around 200 images for model training, from the images I augmented from the test cases we were given
Note: The scaler is a StandardScaler from scikit-learn - which is used to standardize features
'''

import cv2
import numpy as np
import joblib
import sys
#from google.colab.patches import cv2_imshow 

coin_types = {0: 25, 1: 1, 2: 10, 3: 5}

def main():

    #Read the image file name from standard input
    #img = cv2.imread('/Users/feliciadrysdale/Desktop/Project1_CV/16.png', cv2.IMREAD_GRAYSCALE) / 255.0

    #Read the image file name from standard input
    image_file_name = sys.stdin.readline().strip()

    #Read the image
    img = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE) / 255.0

    #Get image dimensions
    height, width = img.shape[:2]

    #Make image quality worse by resizing
    scale_percent = 10
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    #Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(resized_img, (3, 3), 0)
    resized_img = (blurred_image * 255).astype(np.uint8)

    mean_intensity = np.mean(resized_img)
    std_intensity = np.std(resized_img)

    #Set dynamic thresholds based on the image statistics
    lower_threshold = max(0, mean_intensity - 3.4 * std_intensity)
    upper_threshold = min(255, mean_intensity + 3.4 * std_intensity)

    #Detect edges using Canny
    edges = cv2.Canny(resized_img, lower_threshold, upper_threshold)

    #Initialize parameters, max and mins for diameters
    thr_min = 16**2
    thr_max = 26**2
    thr_votes = 0.2

    #Make range for radii so they don't exceed coins
    radii_range = np.around(np.linspace(8, 13, num=50), decimals=2)

    #Detect circles and classify them
    count, circle_dict = detect_circles(resized_img, edges, radii_range, thr_min, thr_max, thr_votes)
    
    #Load the model and scaler
    model = joblib.load('svm_model_999.pkl')
    scaler = joblib.load('scaler_model_999.pkl')

    '''HERE IS WHERE I USED MACHINE LEARNING AND TRAINED SVM '''

    #Loop through the detected circles and classify them
    for (x, y), radius in circle_dict.items():

        #features
        area = np.pi * radius ** 2
        diameter = 2 * radius 
        
        #Collect features: radius, area, diameter
        features = np.array([radius, area]).reshape(1, -1)
        
        #Scale the features
        features_scaled = scaler.transform(features)
        
        #Predict the coin type
        predicted_class = model.predict(features_scaled)
        circle_dict[(x, y)] = {'radius': radius, 'type': predicted_class[0]}

    '''END MACHINE LEARNING PART'''

    #Print and display results
    print(f"{len(circle_dict)}")
    for (x, y), details in circle_dict.items():
        coin_value = coin_types[details['type']]
        print(f"{x} {y} {coin_value}")

#Checking if a circle is new
def new_circle(cx, cy, detected_circles, min_dist=11):
    for circle in detected_circles:
        prev_cx, prev_cy, prev_radius = circle
        if (abs(cx - prev_cx) < min_dist) and (abs(cy - prev_cy) < min_dist):
            return False
    return True

#Try different radii for validation
def check_radius(cx, cy, radii_range, edges, width, height, detected_circles, min_dist):
    for radius in radii_range:
        theta = np.linspace(0, 2 * np.pi * 0.99, 99)
        xs = (cx + radius * np.cos(theta)).astype(int)
        ys = (cy + radius * np.sin(theta)).astype(int)

        #Create a mask for valid points
        valid_mask = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)

        #Count valid edge points
        num_circle_points = np.sum(valid_mask & (edges[ys[valid_mask], xs[valid_mask]] > 127))

        #If enough points lie on the circle, consider it detected
        if num_circle_points > 49:
            #Check if the circle is new before counting
            if new_circle(cx, cy, detected_circles, min_dist):
                return radius
    return None

def detect_circles(resized_img, edges, radii_range, thr_min, thr_max, thr_votes):

    #Count coins, list to store detected circles, dictionary to store circle information
    count = 0
    detected_circles = []  
    circle_dict = {}
    min_dist = 11

    #Create list of edge points
    #NOTE: This is from class lecture!
    edge_pt = []
    for i in range(resized_img.shape[0]):
        for j in range(resized_img.shape[1]):
            if edges[i, j] > 127: 
                edge_pt.append((i, j))

    #Initialize the votes array
    votes = np.zeros((resized_img.shape[0], resized_img.shape[1]), np.int64)

    #Compute votes between edge pairs
    #NOTE: This is from class lecture!
    for i in range(len(edge_pt)):
        y1, x1 = edge_pt[i]
        for j in range(i + 1, len(edge_pt)):
            y2, x2 = edge_pt[j]
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if d < thr_min or d > thr_max:
                continue
            #Calculate potential circle center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 0 <= cx < resized_img.shape[1] and 0 <= cy < resized_img.shape[0]:
                votes[cy, cx] += 1

    #Normalize votes to the range [0, 1]
    #if votes dont equal zero
    if np.amax(votes) != 0:
        votes = votes / np.amax(votes)
    else:
        votes = np.zeros((resized_img.shape[0], resized_img.shape[1]), np.int64)

    #Find and validate circle centers based on votes
    for cy in range(resized_img.shape[0]):
        for cx in range(resized_img.shape[1]):
            if votes[cy, cx] > thr_votes:
                #Check radius
                radius = check_radius(cx, cy, radii_range, edges, resized_img.shape[1], resized_img.shape[0], detected_circles, min_dist)
                if radius is not None:
                    #Store the detected circle
                    detected_circles.append((cx, cy, radius))  
                    count += 1
                    #Store detected circle in a dictionary - multiply by 10 for image size reduction
                    circle_dict[((cx * 10), (cy * 10))] = radius
                    #print(f"Detected valid circle at ({cx}, {cy}), Radius: {radius}")

    return count, circle_dict

main()
