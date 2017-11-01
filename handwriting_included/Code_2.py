# -*- coding: utf-8 -*-

from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import random
import os
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
from sys import getdefaultencoding
import sys
import random
import matplotlib.pyplot as plt
import Code_1 as code
def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck
d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")
def imsave(fname, arr, vmin=None, vmax=None, cmap='gray', format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

NumberOfImage = random.randint(1,5)
ImageWidth = NumberOfImage * 56
ImageWidth  = ImageWidth + 4

Dict = {'অ': 0,
            'আ': 1,
            'ই': 2,
            'ঈ': 3,
            'উ': 4,
            'ঊ': 5,
            'ঋ': 6,
            'এ': 7,
            'ঐ': 8,
            'ও': 9,
            'ঔ': 10,
            'ক': 11,
            'খ': 12,
            'গ': 13,
            'ঘ': 14,
            'ঙ': 15,
            'চ': 16,
            'ছ': 17,
            'জ': 18,
            'ঝ': 19,
            'ঞ': 20,
            'ট': 21,
            'ঠ': 22,
            'ড': 23,
            'ঢ': 24,
            'ণ': 25,
            'ত': 26,
            'থ': 27,
            'দ': 28,
            'ধ': 29,
            'ন': 30,
            'প': 31,
            'ফ': 32,
            'ব': 33,
            'ভ': 34,
            'ম': 35,
            'য': 36,
            'র': 37,
            'ল': 38,
            'শ': 39,
            'ষ': 40,
            'স': 41,
            'হ': 42,
            'ড়': 43,
            'ঢ়': 44,
            'য়': 45,
            'ঃ':46,
            'ৎ' : 47,
            'ং' : 48
            }
background = Image.new('L', (564, 64), (0))
bg_w, bg_h = background.size
img_h = 64
text = "এক কলম সকল"

splitwords=[]
PrevInd=0
for i in range(len(text)):
    if text[i]==' ':
        splitwords.append(text[PrevInd:i])
        PrevInd =i+1
        # for i in range(PrevInd,i,3):
    if i==len(text)-1:
        splitwords.append(text[PrevInd:i+1])

Flag=0
Start = 0
for i in splitwords:
    if(Flag==0):
        text=i
        img_1 = code.paint_text(text, 128, 64)
        imsave('test_image.png', img_1)
        im = Image.open('test_image.png')
        offset = (Start, (bg_h - img_h) / 2)
        background.paste(im, offset)
        Start += 128
        Flag=1
    else:
        convertedarray=[]
        for j in range(0, len(i), 3):
            convertedarray.append(Dict[i[j:j + 3]]+1)
        print(convertedarray)
        for j in range(len(convertedarray)):
            charnumber = convertedarray[j]
            ImageName = 'Images/' + str(charnumber)
            for filename in os.listdir(ImageName):
                ImageName += "/"
                ImageName += filename
                break
            img = Image.open(ImageName)

            # img=Image.open('a.png')
            img = img.resize((56, 56), Image.ANTIALIAS)
            offset = (Start, (bg_h - img_h) / 2)
            background.paste(img, offset)

            Start += 56
        Flag=0

plt.imshow(background,cmap='gray')
plt.show()
