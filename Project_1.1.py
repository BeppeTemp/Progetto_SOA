import cv2
import base64

import numpy as np

from PIL import Image
from time import gmtime, strftime

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

def from_json_to_list(json):
    people = json['people']
    l = [[[0.0, 0.0]]*int(len(people[0]['pose_keypoints_2d'])/3)
         for i in range(int(len(people)))]
    for x in range(len(people)):
        person_point = people[x]['pose_keypoints_2d']
        for y in range(int(len(person_point)/3)):
            z = y*3
            l[x].insert(y, [person_point[z], person_point[z+1]])
    return l
def convertToImage(sbase64):
    return base64.b64decode(sbase64)
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
def crop(l_crop=[]):
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
def processoScript_final(row):
    parts = from_json_to_list(row)
    result=open("temp.jpg", "wb")
    result.write(convertToImage(row["image"]))
    img=cv2.imread('temp.jpg')
    result.close()

    if img is not(None):
        img2, l_crop = bounding_box(img, parts)
        imgcrop = crop(l_crop)

        for x in range(len(imgcrop)):
            dest="temp.jpg"
            imgcrop[x].save(dest)
            f_binary=open(dest,'rb')
            row["image"] = str(base64.b64encode(f_binary.read()))[1:].replace("'","")
    
    return row
    
if __name__ == "__main__":
    conf = SparkConf().setAppName("SOA Project").setMaster("yarn")
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    #Caricamento dataset
    data = spark.read.json("Arienzo-Giordano-Scotti/DataSet.json")

    #Estrapolazione dati
    source = data.select("id","image","people","version")
    
    #Elaborazione del dataset
    result = source.rdd.map(lambda row: (processoScript_final(row.asDict())))

    #Definizione schema
    person = StructType([ \
        StructField("person_id",StringType(),True), \
        StructField("pose_keypoints_2d",StringType(),True), \
        StructField("face_keypoints_2d",StringType(),True), \
        StructField("hand_left_keypoints_2d", StringType(), True), \
        StructField("hand_right_keypoints_2d", StringType(), True), \
        StructField("pose_keypoints_3d", StringType(), True), \
        StructField("face_keypoints_3d", StringType(), True), \
        StructField("hand_left_keypoints_3d", StringType(), True), \
        StructField("hand_right_keypoints_3d", StringType(), True), \
    ])
    schema = StructType([ \
        StructField("id",StringType(),True), \
        StructField("image",StringType(),True), \
        StructField("person",ArrayType(person),True), \
        StructField("version", StringType(), True), \
    ])

    #Creazione del file dei result
    result = spark.createDataFrame(result, schema)

    #Filtraggio file dei result
    result = result.select("image")

    #Salvataggio file dei result
    result.repartition(1).write.format("json").save("Arienzo-Giordano-Scotti/result.json")
    #result.show()

    print("Esecuzione completata")
    spark.stop()