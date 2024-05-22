import cv2
import matplotlib.pyplot as plt

imgcv = cv2.imread( 'app/static/floor_img.jpg' )

plt.imshow(imgcv)
plt.show()