import numpy as np
import PIL


Images = np.zeros((250, 150, 150))
Labels = np.zeros((1, 250))
for i in range(250):
    str1 = './list/' + str(i) + '.png'

    # open the picture
    image = PIL.Image.open(str1)
    # turn into (grayscale)
    image = image.convert('L')

    # resize into (150,150)
    image = image.resize((150, 150))
    
    # check
    if image.size != (150, 150):
        print(i)

    str2 = './list_new/' + str(i) + '.png'
    image.save(str2)

    # store the images in 250*150*150 numpy array
    imageArray = np.array(image)

    Images[i, :, :] = imageArray

    # store true value in 1*250 numpy array
    Labels[:, i] = int(i/10)

# save Images and Labels to npy
np.save('Images', Images)
np.save('Labels', Labels)
