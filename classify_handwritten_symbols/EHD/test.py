from PIL import Image
import numpy as np


Images = np.load('Images-old.npy')
Labels = np.load('labels-old.npy')
print(Images)
print(Labels)


for i in range(Images.shape[0]):
    test_img = Images[i, :, :]
    img = Image.fromarray(test_img)
    if img.mode == 'F':
        img = img.convert('L')
    str2 = './list_new/' + str(i) + '.png'
    img.save(str2)

    if test_img.shape != (150, 150):
        print('shape is wrong')

    img_l = img.convert('L')
    if img != img_l:
        print('L is wrong')

