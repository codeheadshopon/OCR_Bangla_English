# coding=utf8


import random
from sys import getdefaultencoding

import sys


def GenerateRandomBanglaCharsOnly(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    C = 0
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    stringlist = []

    for i in range(num_words):
        NumberofAlphabets = random.randint(1, 7)
        string = ""
        for j in range(NumberofAlphabets):
            Character = random.randint(0, 45)
            string += charlist[Character]
        stringlist.append(string)
    return stringlist


def GenerateRandomBanglaCharsWithModifier(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character
    b = (chars[3:6] + chars[6:9])  # The "Ri Character consits of two different charcters
    modifiers = "া ে ি ী ু"
    charlist = []  # I have the chars in a list in specific index
    modlist = []  # Same for the modifiers
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    for i in range(0, 19, 4):
        modlist.append(modifiers[i * 3:(i + 1) * 3])
    modlist.append(b)
    stringlist = []
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    for i in range(num_words):
        string = ""
        NumberOfAlphabet = random.randint(1, 7)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier = random.randint(0, 1)
            if (WithModifier_or_WithNoModifier):

                indexofchar = random.randint(0, 32)
                string += (charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier = random.randint(0, 5)
                string += (charlist[indexofchar] + modlist[indexofmodifier])
        stringlist.append(string)
    return stringlist


def GenerateRandomBanglaCharsWithModifierAndPunctuation(num_words):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character
    b = (chars[3:6] + chars[6:9])  # The "Ri Character consits of two different charcters
    modifiers = "া ে ি ী ু"
    puncuation = ".,;\" :!"
    punc = []
    for i in puncuation:
        punc.append(i)
    charlist = []  # I have the chars in a list in specific index
    modlist = []  # Same for the modifiers
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    for i in range(0, 19, 4):
        modlist.append(modifiers[i * 3:(i + 1) * 3])

    modlist.append(b)

    Total = charlist + modlist + punc

    stringlist = []
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    for i in range(num_words):
        string = ""
        NumberOfAlphabet = random.randint(1, 3)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier = random.randint(0, 1)
            if (WithModifier_or_WithNoModifier):

                indexofchar = random.randint(0, 32)
                string += (charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier = random.randint(0, 5)
                string += (charlist[indexofchar] + modlist[indexofmodifier])

        punrand = random.randint(0, 5)
        string += puncuation[punrand]

        NumberOfAlphabet = random.randint(1, 3)
        for j in range(NumberOfAlphabet):

            WithModifier_or_WithNoModifier = random.randint(0, 1)
            if (WithModifier_or_WithNoModifier):

                indexofchar = random.randint(0, 32)
                string += (charlist[indexofchar])
            else:
                indexofchar = random.choice(allowed_values)
                indexofmodifier = random.randint(0, 5)
                string += (charlist[indexofchar] + modlist[indexofmodifier])

        stringlist.append(string)
    return stringlist


def GenerateRandomEnglishLowerChars(num_words):
    stringlist = []
    for i in range(num_words):
        string = ""
        NumberofChars = random.randint(1, 7)
        for j in range(NumberofChars):
            ind = random.randint(0, 25)
            string += chr(97 + ind)

        stringlist.append(string)
    return stringlist


def GenerateRandomEnglishUpperChars(num_words):
    stringlist = []
    for i in range(num_words):
        string = ""
        NumberofChars = random.randint(1, 7)
        for j in range(NumberofChars):
            ind = random.randint(0, 25)
            string += chr(65 + ind)

        stringlist.append(string)
    return stringlist


def Jointchars():
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    chars = "খ্র"  # For generating the "Ri" Character

    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    str = "ন্ত"
    hosh = str[3:6]
    char = ""
    jointchars = []

    allowed_for_Po = [31, 38, 41, 39]  # Allowed Character List
    for ind in allowed_for_Po:
        char = charlist[ind] + hosh + "প"
        jointchars.append(char)

    allowed_for_Do = [12, 13, 16, 23, 25, 30, 38, 39]
    for ind in allowed_for_Do:
        char = charlist[ind] + hosh + "ড"
        jointchars.append(char)

    allowed_for_To = [11, 12, 13, 16, 21, 25, 26, 31, 33, 35, 38, 39, 40, 41]
    for ind in allowed_for_To:
        char = charlist[ind] + hosh + "ট"
        jointchars.append(char)

    allowed_for_cho = [16, 18, 20, 21, 24, 25, 31, 38, 39, ]
    for ind in allowed_for_cho:
        char = charlist[ind] + hosh + "চ"
        jointchars.append(char)

    allowed_for_go = [13, 28, 29, 30]
    for ind in allowed_for_go:
        char = charlist[ind] + hosh + "গ"
        jointchars.append(char)

    allowed_for_ko = [11, 15, 38, 40, 41]
    for ind in allowed_for_ko:
        char = charlist[ind] + hosh + "ক"
        jointchars.append(char)

    allowed_for_bo = [11, 12, 13, 14, 17, 18, 21, 23, 25, 26, 27, 28, 29, 30, 33, 35, 38, 39, 40, 41]
    for ind in allowed_for_bo:
        char = charlist[ind] + hosh + "ব"
        jointchars.append(char)

    allowed_for_to = [11, 26, 30, 31, 35, 39]
    for ind in allowed_for_to:
        char = charlist[ind] + hosh + "ত"
        jointchars.append(char)

    allowed_for_do = [13, 28, 30, 33, 35, 38]
    for ind in allowed_for_do:
        char = charlist[ind] + hosh + "দ"
        jointchars.append(char)

    allowed_for_no = [11, 13, 14, 16, 26, 28, 29, 30, 31, 35, 39, 41, 42]
    for ind in allowed_for_no:
        char = charlist[ind] + hosh + "ন"
        jointchars.append(char)

    allowed_for_ro = list(range(11, 46))
    for ind in allowed_for_ro:
        char = charlist[ind] + hosh + "র"
        jointchars.append(char)

    allowed_for_zo = list(range(11, 46))
    for ind in allowed_for_zo:
        char = charlist[ind] + hosh + "য"
        jointchars.append(char)

    allowed_for_bo = list(range(11, 46))
    for ind in allowed_for_bo:
        char = charlist[ind] + hosh + "ব"
        jointchars.append(char)

    char = charlist[11] + hosh + "ষ"
    jointchars.append(char)

    char = charlist[18] + hosh + charlist[18]
    jointchars.append(char)

    char = "ন" + hosh + charlist[18]
    jointchars.append(char)

    char = "ষ" + hosh + "ঠ"
    jointchars.append(char)

    charlist = ["্যা", "ঙ্গ", "প্ল", "ন্স", "ল্ল", "ন্ট", "ন্ধ", "চ্ছ"]
    for i in charlist:
        jointchars.append(i)

    return jointchars


def getTotalData():
    #
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং"
    charlist = []  # I have the chars in a list in specific index
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])

    chars = "খ্র"  # For generating the "Ri" Character
    b = (chars[3:6] + chars[6:9])  # The "Ri Character consits of two different charcters

    modifiers = "ঁ া ে ি ী ু ো ৌ ূ ৗ "
    puncuation = ".,;\"-\' :"
    punc = []
    for i in puncuation:
        punc.append(i)
    charlist = []  # I have the chars in a list in specific index
    modlist = []  # Same for the modifiers
    for i in range(len(banglachars) / 3):
        charlist.append(banglachars[i * 3:(i + 1) * 3])
    C = 0
    for i in range(0, 39, 4):
        modlist.append(modifiers[i:i + 3])

    lowerchar = []
    upperchar = []

    for i in range(97, 123):
        lowerchar.append(chr(i))
    for i in range(65, 91):
        upperchar.append(chr(i))

    Joint = Jointchars()

    Total = charlist + modlist + Joint + punc + lowerchar + upperchar
    return Total


def Makestr():
    '''
        Ranges:
        Character = 0-45
        Modifier = 46-50
        Joint = 51 - 172/173
        Punctuation = 173/174 - 178/179
        LowerChar= 179/180-204/205
        UpperChar = 205/206-230/231
    '''
    Total = getTotalData()
    allowed_values = list(range(11, 46))
    allowed_values.remove(15)
    allowed_values.remove(20)

    String = ""
    BanglaOrEnglishOrPunc = random.randint(0, 1)
    if (BanglaOrEnglishOrPunc == 0):
        NormalOrPunc = random.randint(0, 3)  # Punctuation or Normal Character
        if (NormalOrPunc == 3):
            PuncInd = random.randint(174, 179)
            String = Total[PuncInd]
        else:
            NormalOrJoinOrModifier = random.randint(0, 2)
            if (NormalOrJoinOrModifier == 0):
                IndexofNormal = random.randint(0, 45)
                String = Total[IndexofNormal]
            elif (NormalOrJoinOrModifier == 1):
                IndexofNormal = random.choice(allowed_values)
                IndexofMod = random.randint(46, 50)
                String = Total[IndexofNormal] + Total[IndexofMod]
            else:
                IndexofJoint = random.randint(51, 173)
                String = Total[IndexofJoint]
    elif (BanglaOrEnglishOrPunc == 1):
        NormalorPunc = random.randint(0, 3)
        if (NormalorPunc == 3):
            PuncInd = random.randint(174, 179)
            String = Total[PuncInd]
        else:
            UpperOrLower = random.randint(0, 1)
            if (UpperOrLower):
                IndexUp = random.randint(180, 205)
                String = Total[IndexUp]
            else:
                IndexDown = random.randint(206, 231)
                String = Total[IndexDown]
    # else:
    #     PuncInd = random.randint(173, 178)
    #     String = Total[PuncInd]

    return String


import time


def decodetheData(string):
    Index = {}
    C = 0
    Total = getTotalData()
    for i in Total:
        Index[i] = C
        C += 1
    i = 0
    labels = []
    while i < len(string):
        isJoint = 0
        isChar = 0
        isNorm = 0
        if (i + 9 <= len(string)):
            ifjointornot = string[i:i + 9]
            Flag = 0
            for x in Total:
                if (ifjointornot == x):
                    isJoint = 1
                    i += 9
                    labels.append(Index[x])
                    break

        if (isJoint == 0):
            if (i + 3 <= len(string)):
                ifcharornot = string[i:i + 3]
                Flag = 0
                for x in Total:
                    if (ifcharornot == x):
                        Flag = 1
                        isChar = 1
                        i += 3
                        labels.append(Index[x])
                        break
        if (isJoint == 0 and isChar == 0):
            for x in Total:
                if (x == string[i]):
                    i += 1
                    labels.append(Index[x])
                    isNorm = 1
                    break
        if (isJoint == 0 and isChar == 0 and isNorm == 0):
            i += 1
            # print(i, " ", len(string))
            # print(labels)

    return labels


def CombinedDataset(num_words):
    strings = []
    for i in range(num_words):
        numberofchars = random.randint(3, 12)
        string = ""
        for i in range(numberofchars):
            string += Makestr()
        strings.append(string)
    # decodetheData(string)
    return strings


def Bangla():
    file = "texts/list_"
    wordlist = []
    for filenumber in range(1, 5):
        with open(file + str(filenumber) + ".txt", 'rt') as f:
            for line in f:
                if(len(line)<=60):
                    if(line[len(line)-1]=='\n'):
                        line=line[0:len(line)-1]
                    wordlist.append(line)

    wordlist = list(set(wordlist))
    # print(len(wordlist))

    with open("texts/words.txt", 'rt') as f:
        for line in f:
            C = 0
            ind = 0
            for i in line:
                if (i == '|'):
                    C += 1
                if (C == 2):
                    ind += 1
                    break
                ind += 1
            string = line[ind:len(line)]
            # print(string)
            arr = string.split(" ")
            for j in arr:
                if(j!=""):
                    if (len(j) <= 60):
                        if (j[len(j) - 1] == '\n'):
                            j = j[0:len(j) - 1]
                        wordlist.append(j)

    # print(len(wordlist))
    wordlist = list(set(wordlist))
    return wordlist


def English():
    engwords = []
    with open("texts/wordlist_bi_clean.txt", 'rt') as f:
        for line in f:
            arr = line.split(" ")
            for j in arr:
                if(j!=""):
                    if(len(j)<=12):
                        if(j[len(j)-1]=='\n'):
                            j=j[0:len(j)-1]
                        engwords.append(j)
    with open("texts/wordlist_mono_clean.txt", 'rt') as f:
        for line in f:
            arr = line.split(" ")
            for j in arr:
                if(j!=""):
                    if(len(j)<=12):

                        if (j[len(j) - 1] == '\n'):
                            j = j[0:len(j) - 1]

                        engwords.append(j)
    engwords = list(set(engwords))
    return engwords


def makeData(num_words,type):
    Bang = Bangla()

    Bang.remove("")

    Eng = English()
    Eng.remove("")
    Stringlist = []
    puncuation = ".,;\"-\' :"
    for word in range(num_words):
        string = ""
        if(type=="singleword"):
            numword = 1
        else:
            numword = random.randint(1, 4)
        for add in range(numword):
            BangEngPun = random.randint(0, 2)
            if (BangEngPun == 0):
                string += Bang[random.randint(0, len(Bang) - 1)]
            elif (BangEngPun == 1):
                string += Eng[random.randint(0, len(Eng) - 1)]

            if (random.randint(0, 2) == 1):
                string += puncuation[random.randint(0, len(puncuation) - 1)]

            string += " "

        r=random.randint(0, 2)
        if (r==0):
            string += "."
        elif(r==1):
            string += "?"


        Stringlist.append(string)
        # print(string)
        # print(len(string))

    return Stringlist


def basiWords():
    Bang = Bangla()
    Bang.remove("")
    basicWords=[]
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং"
    chars = []
    for i in range(len(banglachars)):
        chars.append(banglachars[i:i+3])
    ItsKo=0
    for i in Bang:

        NotBasic=0
        for j in range(0,len(i),3):

            char = i[j:j+3]
            # print(char)
            Flag=0
            for ch in chars:

                if char==ch:
                    Flag=1
                    break
            if(Flag==0):
                NotBasic=1
                break
        if(not NotBasic):
            if((len(i)/3)<4):
                # print(i)
                basicWords.append(i)

    return basicWords

def newDataset(num_words,type):
    Bang=basiWords()
    text=""
    stringlist=[]
    for totalwords in range(num_words):
        text=""
        numwords = random.randint(2, 4)
        for i in range(numwords):
            index= random.randint(0,len(Bang)-1)
            text+=Bang[index]
            text+=" "
        # print(text)

        stringlist.append(text)
    return stringlist
def labelingNewDataset(text):

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
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং"

    chars = []
    for i in range(0,len(banglachars),3):
        chars.append(banglachars[i:i+3])
    splitwords=[]
    PrevInd=0
    for i in range(len(text)):
        if text[i] == ' ':
            splitwords.append(text[PrevInd:i])
            PrevInd = i + 1
            # for i in range(PrevInd,i,3):
        if i == len(text) - 1:
            splitwords.append(text[PrevInd:i + 1])
    labels=[]
    WordCounter=0
    for i in splitwords:
        WordCounter+=1
        for j in range(0,len(i),3):
            labels.append(Dict[i[j:j+3]])
        if(WordCounter<len(splitwords)):
            labels.append(49)
    return labels
def decodeNewDataset(labels):
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং "
    chars = []
    for i in range(0,len(banglachars),3):
        chars.append(banglachars[i:i + 3])

    # print(labels)
    text=""
    # print(labels)
    for i in labels:

        if(i!=50):
            text+=chars[i]
    return text
# print(newDataset(10,"ab"))
