import base64, json, os

img_path = "IMAGE/"
json_path = "JSON/"
result_path = "RESULT/"

def convertoToBase64(fname):
    with open(fname, "rb") as img_file:
        return base64.b64encode(img_file.read())

def convertToImage(sbase64):
    return base64.b64decode(sbase64)

def createSingleJSON(fname, img_list):
    #Apro il file JSON iniziale e lo converto in oggetto python
    with open(json_path + fname + "_keypoints.json", "r") as json_file:
        j = json.loads(json_file.read())

    #Aggiungo il campo image
    if fname+".jpg" in img_list:
        j["image"] = str(convertoToBase64(img_path + fname+".jpg"))[1:].replace("'","")

    return j

def createJSON(img_list, json_list):
    with open(result_path + "DataSet_Grezzo_.json", "w") as data_set:
        i = 0

        for f in json_list:
            i = i + 1
            if (i % 10000 == 0):
                print("Processate " + str(i) + " immagini.")

            f = f.replace('_keypoints.json','')
            j = createSingleJSON(f, img_list)
            j["id"] = f
            data_set.write(json.dumps(j))

        data_set.close()


print("Avvio programma e ordinamento liste")
img_list = sorted(os.listdir(img_path))
json_list = sorted(os.listdir(json_path))
print("Ordinamento completato")

createJSON(img_list, json_list)

print("Qualcosa è esploso ?")