import skimage.io as io
import os
import numpy as np
from skimage.transform import resize
from skimage.transform import rotate

directory = "/vagrant/data/train"
counter = 0
xvalues = []
yvalues = []

for root, subFolders, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".jpg"): 
            image = io.imread(os.path.join(root, filename))
            x, y, z = image.shape
            if x >= y:
                print "x = %d, y = %d" % (x, y)
            xvalues.append(x)
            yvalues.append(y)
            rescaled_img_name = "RESCALED_" + os.path.splitext(os.path.basename(filename))[0] + ".jpg"
            rescaled_img_path = os.path.join(root, rescaled_img_name)
            rescaled_img = resize(image, (760, 1280))
            io.imsave(rescaled_img_path, rescaled_img)
            rotated_image = rotate(rescaled_img, 180, resize=False)
            rotated_img_name = "ROTATED_" + os.path.splitext(os.path.basename(filename))[0] + ".jpg"
            rotated_img_path = os.path.join(root, rotated_img_name)
            io.imsave(rotated_img_path, rotated_image)
            counter += 1
            if counter % 500 == 0:
                print counter

print "Min x = %d" % min(xvalues)
print "Max x = %d" % max(xvalues)

print "Min y = %d" % min(yvalues)
print "Max y = %d" % max(yvalues)

print "Mean x = %f" % np.mean(xvalues)
print "Mean y = %f" % np.mean(yvalues)

print "Number of images processed: %d"  % counter