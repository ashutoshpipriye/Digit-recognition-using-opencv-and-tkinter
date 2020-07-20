from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
def select_image():
	global panelA, panelB
	path = filedialog.askopenfilename()
	if len(path) > 0:
		img = cv2.imread(path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		clf, pp = joblib.load("digits_cls.pkl")
		im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
		ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
		ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(ctr) for ctr in ctrs]

		im = im_gray.copy()

		im_gray = Image.fromarray(im)
		im_gray = ImageTk.PhotoImage(im_gray)

		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
			
			panelB = Label(image=im_gray)
			for rect in rects:
				cv2.rectangle(im_gray, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
				# cv2.rectangle(im_gray, (0,0), (220, 220), (0, 255, 0), 3) 

				# Draw the rectangles
				leng = int(rect[3] * 1.6)
				pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
				pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
				roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
				# Resize the image
				roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
				roi = cv2.dilate(roi, (3, 3))
				# Calculate the HOG features
				roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
				roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
				nbr = clf.predict(roi_hog_fd)
				im_gray = cv2.putText(im_gray, str(int(nbr[0])),(rect[0], rect[1]),*font, 2, (0, 255, 255), 3)
				panelB.image = im_gray
			panelB.pack(side="right", padx=10, pady=10)
root = Tk()
panelA = None
panelB = None
root.resizable(0, 0)
root.title("Digit Recognition")
canvas1 = Canvas(root, width = 300, height = 100, bg = 'black', relief = 'raised')
canvas1.pack()
btn = Button(text="      Select an image     ", command=select_image, bg='white', fg='black', font=('helvetica', 12, 'bold'))
canvas1.create_window(150,50,window=btn)
root.mainloop()