In this project, I applied machine learning techniques to classify different types of coins by processing images. 
I manually reviewed and prepared the test images by erasing the background in each image and replacing it with different backgrounds,leaving only the coin visible. I needed more data to improve accuracy, so I wrote a script that randomly augmented the data that I had and saved it as a new image, thus I had more data to train the SVM model. 

The augmentation involved a random selection of blurring, flipping, adding noise, lightening and darkening the images. After I created more data, I used Canny edge detection, to extract important features such as radius and area (thousandths decimal) from the contours of the coins. 

These features were then used to train a Support Vector Machine (SVM) classifier to differentiate between different coin types: quarters, pennies, dimes, and nickels. The images were resized, scaled, and blurred to improve the model's performance. 

By manually preparing the dataset and focusing on the key features of each coin, I was able to train the model, achieving an accuracy of 93.02%.

Note: I ended up having around 200 images for model training, from the images I augmented from the test cases we were given
Note: The scaler is a StandardScaler from scikit-learn - which is used to standardize features