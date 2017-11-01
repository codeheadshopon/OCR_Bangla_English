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

d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")


def speckle(img):
    severity = np.random.uniform(0, 0.0)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, fontsize, rotate=False, ud=True, multi_fonts=False):
    newtext = ""
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং"
    # text = "নড়চড়"
    chars = []
    for i in range(0,len(banglachars),3):
        chars.append(banglachars[i:i+3])
    for i in range(0,len(text),3):
        ch=text[i:i+3]
        itsoke= 1
        for j in chars:
            if j==ch:
                itsoke = 0
        if(itsoke):
            print("Something fishy with this text")
            print(text)
    # text="অ"
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    import random

    FlagBlack = random.randint(0, 4)
    FlagBlack = 2
    with cairo.Context(surface) as context:
        if (FlagBlack == 2):
            context.set_source_rgb(0, 0, 0)  # White
        else:
            context.set_source_rgb(1, 1, 1)  # White

        context.paint()
        # this font list works in CentOS 7
        if multi_fonts:
            fonts = ['Solaimanlipi', 'Siyamrupali', 'kalpurush', 'Lohit', 'prothoma']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
            context.select_font_face('Siyamrupali', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        import random

        context.set_font_size(fontsize)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            while box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                idx=len(text)-1
                for i in range(len(text)-1,0,-1):
                    # print(text[i])
                    if text[i]==" ":
                        idx=i
                        break
                text=text[0:idx]
                box = context.text_extents(text)

            box = context.text_extents(text)

            # raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate


        max_shift_x = w - box[2]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            rando= np.random.randint(0, int(max_shift_y))

            top_left_y =  rando
        else:
            if fontsize>40:
                top_left_y = h // 6
            elif fontsize>35:
                top_left_y = h // 4
            elif fontsize>30:
                top_left_y = h // 3
            else:
                top_left_y = h // 2


        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        if (FlagBlack == 2):
            context.set_source_rgb(1, 1, 1)
        else:
            context.set_source_rgb(0, 0, 0)

        # print(text)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    # a = speckle(a)

    # a = a.astype(np.float32) / 255
    # a = np.expand_dims(a, 0)
    # if rotate:
    #     a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)


    return a
# NumberOfImage = random.randint(1,5)
# ImageWidth = NumberOfImage * 56
# ImageWidth  = ImageWidth + 4
#
# background = Image.new('RGB', (ImageWidth+128, 64), (0, 0,0,255))
#
#
#
#
# img_1= Image.open('newt.png').convert('L')
# # im1=np.asarray(img)
# # print("oke",im1.shape)
# # im1=np.resize(img_1,(64,128))
# # img_1=Image.fromarray(im1)
# # plt.imshow(img,cmap='gray')
# #
# # plt.show()
# # Start = 0
# bg_w, bg_h = img_1.size
# img_h = 64
# # offset = (Start, (bg_h - img_h) / 2)
# # background.paste(img_1, offset)
# # plt.imshow(background)
# # plt.show()
# Start=128
# for i in range(NumberOfImage):
#     charnumber = random.randint(1,84)
#     # ImageName='Images/'+str(charnumber)
#     # for filename in os.listdir(ImageName):
#     #     ImageName+="/"
#     #     ImageName+=filename
#     #     break
#     # img=Image.open(ImageName)
#     img=Image.open('a.png')
#     img = img.resize((56, 56), Image.ANTIALIAS)
#     offset = (Start, (bg_h - img_h) / 2)
#     img_1.paste(img, offset)
#
#     Start+=56
# # background-background.resize()
# plt.imshow(img_1)
# plt.show()
#
#
# #
# # import numpy
# # def PIL2array(img):
# #     im=np.asarray(img)
# #     print(im.shape)
# #     print(im.shape[1])
# #     img=np.reshape(im,(64,im.shape[1]))
# #     print("New Shape - ",img.shape)
# #     return numpy.array(img.getdata(),
# #                     numpy.uint8).reshape(img.size[1], img.size[0])
# #
# #
# #
# # image_1=np.reshape(image,(64,128))
# # image_2=np.asarray(background)
# # print(image_2.shape)
# # image_2 = np.resize(image_2,(64,image_2.shape[1]))
# #
# #
# # im1=Image.fromarray(image_1)
# # im2=Image.fromarray(image_2)
# #
# # im1.paste(im2,(128,0))
# # plt.imshow(im1)
# # plt.show()
# #
