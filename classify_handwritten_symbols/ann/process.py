import numpy as np
from PIL import Image


Images = np.zeros((250, 150, 150))
Labels = np.zeros((1, 250))
for i in range(250):
    str1 = './DATA/' + str(i+1) + '.png'

    # open the picture
    image = Image.open(str1)
    print(type(image))
    # turn into (grayscale)
    image = image.convert('L')

    # resize into (150,150)
    image = image.resize((150, 150))
    
    # check
    if image.size != (150, 150):
        print(i)

    str2 = './DATA_new/' + str(i+1) + '.png'
    image.save(str2)

    # store the images in 250*150*150 numpy array
    imageArray = np.array(image)

    Images[i, :, :] = imageArray

    # store true value in 1*250 numpy array
    Labels[:, i] = int(i/10)

# save Images and Labels to npy
np.save('Images', Images)
np.save('Labels',Labels)
