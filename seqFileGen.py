import base64, json, os

img_path = "mnt/c/Users/giuse/Desktop/IMAGE/IMAGE/"
json_path = "mnt/c/Users/giuse/Desktop/JSON/JSON/"
result_path = "mnt/c/Users/giuse/Desktop/RESULT/"

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

        '''result = open("result.jpg", "wb")
        result.write(convertToImage(j["image"]))
        result.close()'''

    return j

def createJSON(img_list, json_list):
    with open(result_path + "DataSet.json", "w") as data_set:
        data_set.write("[")
        i = 0

        for f in json_list:
            i = i + 1
            print("Processate " + str(i) + "immagini")

            f = f.replace('_keypoints.json','')
            j = createSingleJSON(f, img_list)
            j["id"] = f
            data_set.write(json.dumps(j))
            if i != len(json_list):
                data_set.write(",")

        data_set.write("]")
        data_set.close()


print("Avvio programma e ordinamento liste")
img_list = sorted(os.listdir(img_path))
json_list = sorted(os.listdir(json_path))
print("Ordinamento completato")

createJSON(img_list, json_list)

print("Qualcosa è esploso ?")


'''#Salvo il JSON finale
with open("RESULT/" + fname + ".json", "w") as json_result:
json_result.write(json.dumps(j))'''