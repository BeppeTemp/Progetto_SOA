import math
import pandas as pd
import numpy as np
from sklearn import metrics
import cv2
import numpy as np
import json
import os
from PIL import Image
import pandas as pd
import math
import time
from time import gmtime, strftime
from sklearn import metrics


def from_json_to_list(path):
    file = open(path)
    data = json.load(file)
    file.close()
    people = data['people']
    l = [[[0.0, 0.0]]*int(len(people[0]['pose_keypoints_2d'])/3)
         for i in range(int(len(people)))]
    """for q in range(len(l)):
        print(len(l[q]))"""
    for x in range(len(people)):
        person_point = people[x]['pose_keypoints_2d']
        for y in range(int(len(person_point)/3)):
            z = y*3
            l[x].insert(y, [person_point[z], person_point[z+1]])
        """while y<(int(len(person_point)-2)):
            e=int(y/3)
            #print(person_point[y])
            l[x][e].insert(0,person_point[y])
            l[x][e].insert(1,person_point[y + 1])
            y=y+3
        y=0"""
    return l


def build_rect(img, point1=[], point2=[], p=""):
    img2 = img

    if (p == "te"):
        const = 1 + 0.7
    if (p == "b"):
        const = 0.35
    if (p == "t"):
        const = 0.6
    if (p == "p"):
        const = 0.8
    if (p == "g"):
        const = 0.4
    if (p == "c"):
        const = 0.4
    if (p == "s"):
        const = 0.8
    if (p == "tes"):
        const = 2

    x1 = int(point1[0])
    x2 = int(point2[0])
    y1 = int(point1[1])
    y2 = int(point2[1])
    # controllo le occlusioni
    if ((x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0)):
        return
    array2 = np.array([(x1, y1), (x2, y2)])
    # prelevo le informazioni sul posizionamento dell'arto
    # calcoliamo l'angolo necessario ad orientare il rettangolo
    info = cv2.minAreaRect(array2)
    # creo le dimensioni della box
    centerx = int(info[0][0])
    centery = int(info[0][1])
    center = (centerx, centery)
    height = int(info[1][0])
    width = int(height*const)
    angle = info[2]
    if (p == "tes"):
        dimensions = (height*2, width)
    elif(p == "p"):
        dimensions = (int(height * 1.5), width)
    else:
        dimensions = (height, width)

    rect2 = (center, dimensions, angle)
    # creo i punti della box
    box = cv2.boxPoints(rect2)
    box = np.int0(box)
    # disegno i punti
    # cv2.drawContours(img,[box],0,(0,255,255),3)
    # taglio con rotazione
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    img_crop = cv2.getRectSubPix(img_rot, rect2[1], center)
    return img, img_crop


def bounding_box(img, l=[]):
    people = len(l)
    l1 = []
    l2 = []
    #l_crop=[[[0.0,0.0]]*25 for i in range(int(len(l)))]

    y = 0
    for x in range(people):
        for y in range(len(l[x])):
            # controllo le occlusioni
            """if(((l[x][y][0]==0)and (l[x][y][1]==0))or((l[x][y+1][0]==0)and (l[x][y+1][1]==0))):
                continue"""
            if((y == 4) or (y == 7) or (y == 8) or (y == 11)):
                continue
            if(y == 1):
                if(not((l[x][y][0] == 0 and l[x][y][1] == 0) or (l[x][8][0] == 0 and l[x][8][1] == 0))):
                    img, img_crop = build_rect(img, l[x][y], l[x][8], "t")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        # cv2.imshow("titolo",img_crop)
                        # cv2.waitKey(0)
                if(not((l[x][y][0] == 0 and l[x][y][1] == 0) or (l[x][5][0] == 0 and l[x][5][1] == 0))):
                    img, img_crop = build_rect(img, l[x][y], l[x][5], "s")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
                if(not((l[x][y][0] == 0 and l[x][y][1] == 0) or (l[x][2][0] == 0 and l[x][2][1] == 0))):
                    img, img_crop = build_rect(img, l[x][y], l[x][2], "s")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
            if((y > 1) and (y < 7)):
                if not((((l[x][y][0] == 0) and (l[x][y][1] == 0)) or ((l[x][y + 1][0] == 0) and (l[x][y + 1][1] == 0)))):
                    img, img_crop = build_rect(img, l[x][y], l[x][y+1], "b")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
            if((y > 8) and (y < 14)):
                if not((((l[x][y][0] == 0) and (l[x][y][1] == 0)) or ((l[x][y + 1][0] == 0) and (l[x][y + 1][1] == 0)))):
                    img, img_crop = build_rect(img, l[x][y], l[x][y + 1], "g")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
            if(y == 17):
                if not((((l[x][y][0] == 0) and (l[x][y][1] == 0)) or ((l[x][y + 1][0] == 0) and (l[x][y + 1][1] == 0)))):
                    img, img_crop = build_rect(img, l[x][y], l[x][y+1], "te")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
                elif not((((l[x][y][0] == 0) and (l[x][y][1] == 0)) or ((l[x][0][0] == 0) and (l[x][0][1] == 0)))):
                    img, img_crop = build_rect(img, l[x][y], l[x][0], "tes")
                    if (img_crop is not (None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
                elif not((((l[x][18][0] == 0) and (l[x][18][1] == 0)) or ((l[x][0][0] == 0) and (l[x][0][1] == 0)))):
                    img, img_crop = build_rect(img, l[x][18], l[x][0], "tes")
                    if (img_crop is not (None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
            if(y == 23):
                if(not((l[x][y][0] == 0 and l[x][y][1] == 0) or (l[x][11][0] == 0 and l[x][11][1] == 0))):
                    img, img_crop = build_rect(img, l[x][y], l[x][11], "p")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
            if (y == 14):
                if(not((l[x][y][0] == 0 and l[x][y][1] == 0) or (l[x][20][0] == 0 and l[x][20][1] == 0))):
                    img, img_crop = build_rect(img, l[x][y], l[x][20], "p")
                    if (img_crop is not(None)):
                        l1.append(img_crop)
                        #cv2.imshow("titolo", img_crop)
                        # cv2.waitKey(0)
        l2.append(l1)
        l1 = []
    return img, l2


def parse_dir(path=''):

    l = []
    l2 = []
    for root, dirs, files in os.walk(path):
        for filename in sorted(files):
            if(filename[0] != '.'):
                filepath = path+'/'+filename
                l.append(filepath)
                l2.append(filename)
                filepath = ''
    return l, l2


def cropvecchia(l_crop=[]):
    """for x in range(len(l_crop)):
        # conc=l[x][0]
        # conc=cv2.resize(l_crop[x][0],(60,50),interpolation=cv2.INTER_AREA)
        for y in range(len(l_crop[x])):
            print(l_crop[x][y].shape)
            l_crop[x][y] = cv2.resize(l_crop[x][y], (60, 50), interpolation=cv2.INTER_AREA)
    imgv = Image.new("RGB", (720, 50))
    for x in range(len(l_crop)):
        for y in range(len(l_crop[x])):
            # PILLOW interpreta i colori in maniera diversa rispetto a cv2 quindi effettuiamo una conversione
            imgconv = cv2.cvtColor(l_crop[x][y], cv2.COLOR_BGR2RGB)
            # Inoltre pillow interpreta le immagini diversamente rispetto a cv2 quindi effettuiamo anche questa conversione
            img1 = Image.fromarray(imgconv)
            # uniamo le immagini con paste
            imgv.paste(img1, (y * 50, 0))"""
    l = []
    for x in range(len(l_crop)):
        hrifa = l_crop[x][0].shape[0]
        if (len(l_crop[x]) >= 5):
            imgv = Image.new("RGB", (1080, hrifa))
            for y in range(len(l_crop[x])):
                # print(l_crop[x][y].shape)
                l_crop[x][y], width = image_resize(
                    l_crop[x][y], 80, 60, hrifa, inter=cv2.INTER_AREA)
                # PILLOW interpreta i colori in maniera diversa rispetto a cv2 quindi effettuiamo una conversione
                imgconv = cv2.cvtColor(l_crop[x][y], cv2.COLOR_BGR2RGB)
                # Inoltre pillow interpreta le immagini diversamente rispetto a cv2 quindi effettuiamo anche questa conversione
                img1 = Image.fromarray(imgconv)
                # uniamo le immagini con paste
                imgv.paste(img1, (y * width, 0))
            l.append(imgv)
    return l


def crop(l_crop=[]):
    """for x in range(len(l_crop)):
        # conc=l[x][0]
        # conc=cv2.resize(l_crop[x][0],(60,50),interpolation=cv2.INTER_AREA)
        for y in range(len(l_crop[x])):
            print(l_crop[x][y].shape)
            l_crop[x][y] = cv2.resize(l_crop[x][y], (60, 50), interpolation=cv2.INTER_AREA)
    imgv = Image.new("RGB", (720, 50))
    for x in range(len(l_crop)):
        for y in range(len(l_crop[x])):
            # PILLOW interpreta i colori in maniera diversa rispetto a cv2 quindi effettuiamo una conversione
            imgconv = cv2.cvtColor(l_crop[x][y], cv2.COLOR_BGR2RGB)
            # Inoltre pillow interpreta le immagini diversamente rispetto a cv2 quindi effettuiamo anche questa conversione
            img1 = Image.fromarray(imgconv)
            # uniamo le immagini con paste
            imgv.paste(img1, (y * 50, 0))"""
    l = []
    for x in range(len(l_crop)):
        if (len(l_crop[x]) >= 5):
            hrifa = l_crop[x][0].shape[0]
            imgv = Image.new("RGB", (1540, 70))
            for y in range(len(l_crop[x])):
                # print(l_crop[x][y].shape)
                #l_crop[x][y],width = image_resize(l_crop[x][y], 80, 60,hrifa*2, inter=cv2.INTER_AREA)
                l_crop[x][y] = cv2.resize(
                    l_crop[x][y], (110, 70), interpolation=cv2.INTER_AREA)
                # PILLOW interpreta i colori in maniera diversa rispetto a cv2 quindi effettuiamo una conversione
                imgconv = cv2.cvtColor(l_crop[x][y], cv2.COLOR_BGR2RGB)
                # Inoltre pillow interpreta le immagini diversamente rispetto a cv2 quindi effettuiamo anche questa conversione
                img1 = Image.fromarray(imgconv)
                # uniamo le immagini con paste
                imgv.paste(img1, (y * 110, 0))
            l.append(imgv)
    return l

# viene utilizzato processoscript


def processo():
    filepath = input("inserisci filepath json: ")
    l = from_json_to_list(filepath)
    imgpath = input("Inserisci imgpath: ")
    img = cv2.imread(imgpath)
    img2, l_crop = bounding_box(img, l)
    imgcrop = crop(l_crop)
    dest = input("Inserisci il percorso di destinazione: ")
    nome = input("inserisci il nome: ")
    dest = dest+"/"+nome+".jpg"
    """sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "/Users/De_Filippo/PycharmProjects/provapy3/mod/EDSR_x4.pb"
    sr.readModel(path)
    sr.setModel("edsr", 4)
    app=imgcrop
    imcv = cv2.cvtColor(np.asarray(app), cv2.COLOR_RGB2BGR)
    result = sr.upsample(imcv)
    cv2.imshow("titttt",result)
    cv2.waitKey(0)"""
    imgcrop.save(dest)


def processoscript(jsonpathdir, imagepathdir, dest):
    l, lname = parse_dir(jsonpathdir)
    l2, l2name = parse_dir(imagepathdir)
    dest2 = dest
    for i in range(len(l2)):
        parts = from_json_to_list(l[i])
        img = cv2.imread(l2[i])
        img2, l_crop = bounding_box(img, parts)
        #cv2.imshow("titt", img2)
        cv2.waitKey(0)
        imgcrop = crop(l_crop)
        for x in range(len(imgcrop)):
            dest = dest+"/"+lname[i][:-15]+"sog_"+str(x)+".jpg"
	    if i % 50000 == 0:
		print(i)
            imgcrop[x].save(dest)
            dest = dest2
	strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


def arraycorr(filepath, str):
    f = pd.read_csv(filepath, sep=';')
    lscore = []
    lposneg = []
    laccuracy = []
    for x in f.iterrows():
        if(str == "01"):
            p1 = math.ceil(int((x[1]['ImgA']) + 1) / 4)
            p2 = a = math.ceil(int((x[1]['ImgB']) + 1) / 4)
            """p1 e p2 calcolati in questo modo sono utili nel caso di CHUCK01"""
        else:
            p1 = int(x[1]['ImgA'])
            p2 = int(x[1]['ImgB'])
            """p1 e p2 calcolati in questo modo sono utili negli altri casi"""
        score = round(float(x[1]['Score']), 2)
        if (score > 0.5):
            laccuracy.append(1)
        else:
            laccuracy.append(0)

        if (p1 == p2):
            if (score > 0.5):
                lscore.append(score)
                lposneg.append(1)  # true positive
            else:
                lscore.append(score)
                lposneg.append(0)  # false negative
        else:
            if (score < 0.5):
                lscore.append(score)
                lposneg.append(1)  # true negative
            else:
                lscore.append(score)
                lposneg.append(0)  # false positive
    l1 = np.array(lscore)
    l2 = np.array(lposneg)

    return l1, l2, laccuracy


def image_resize(image, width=None, height=None, hrif=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = w / h
        diff = hrif-h
        dim = (int(width+(diff*r)), hrif)
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized, width


def score(pathcsv, mode, titds):
    # mode 01 funziona per chuck 01, altrimenti va inserito 02
    sum = 0
    score, true, prediction = arraycorr(pathcsv, mode)
    fpr, tpr, thresholds = metrics.roc_curve(true, score)
    fnr = 1 - tpr
    """for i in range(len(fpr)-1):
        fpr[i]=fpr[i]-(0.05*fpr[i])
        tpr[i] = tpr[i] + (0.05 * tpr[i])"""
    for i in range(1, len(thresholds)):
        sum = sum+thresholds[i]
    app = sum/(len(thresholds)-1)
    print("threshold :", app)
    print(tpr, fpr)
    auc = metrics.auc(fpr, tpr)
    print("auc =", auc)
    aver = metrics.average_precision_score(true, score)
    print("mAP=", aver)
    accuracy = metrics.accuracy_score(true, prediction)
    print("accuracy =", accuracy)
    mse = metrics.mean_squared_error(true, prediction)
    print("mse=", mse)
    plt.plot(fpr, tpr)
    plt.title(titds)
    plt.plot([0, 1], [0, 1])
    # ax[1].set_xlim(0,1)
    # ax[1].set_ylim(0,1)
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.show()


if __name__ == "__main__":
    processoscript(jsonpathdir="/home/soa/Arienzo_Scotti_Giordano/Progetto_SOA/JSON",
                   imagepathdir="/home/soa/Arienzo_Scotti_Giordano/Progetto_SOA/IMAGE", dest="/home/soa/Arienzo_Scotti_Giordano/Progetto_SOA/RESULT")
    # score("/Users/De_Filippo/Desktop/esperimentoMARKET-1501/scorefile.csv","02","MARKET-1501")

    """scorec3,truec3= arraycorr("/Users/De_Filippo/Desktop/esCHUCK-03/scorefile.csv","02")
    scorec1,truec1 = arraycorr("/Users/De_Filippo/Desktop/esperimentoCHUCK-01/scorefile.csv","01")
    scorem, truem = arraycorr("/Users/De_Filippo/Desktop/esperimentoMARKET-1501/scorefile.csv","02")
    fprc3,tprc3,thresholdsc3= metrics.roc_curve(truec3,scorec3)
    fnrc3=1-tprc3
    fprc1, tprc1, thresholdsc1 = metrics.roc_curve(truec1, scorec1)
    fnrc1 = 1 - tprc1
    fprm, tprm, thresholdsm = metrics.roc_curve(truem, scorem)
    fnrm = 1 - tprm
    aucc1=metrics.auc(fprc1,tprc1)
    aucc3 = metrics.auc(fprc3, tprc3)
    aucm = metrics.auc(fprm, tprm)
    print("auc (CHUCK-01) =",aucc1)
    print("auc (CHUCK-03) =", aucc3)
    print("auc (MARKET-1501) =", aucm)
    averc1 = metrics.average_precision_score(truec1, scorec1)
    averc3 = metrics.average_precision_score(truec3, scorec3)
    averm = metrics.average_precision_score(truem, scorem)
    print("mAP (CHUCK-01)=", averc1)
    print("mAP (CHUCK-03)=", averc3)
    print("mAP (MARKET-1501)=", averm)
    fig1,ax1=plt.subplots(3)
    ax1[0].plot(fprc1,tprc1)
    ax1[0].set_title("CUCK-01")
    ax1[1].plot(fprc3, tprc3)
    ax1[1].set_title("CUCK-03")
    ax1[2].plot(fprm, tprm)
    ax1[2].set_title("MARKET-1501")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.show()"""
