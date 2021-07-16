import base64, json, os

def convertoToBase64(fname):
    with open(fname, "rb") as img_file:
        return base64.b64encode(img_file.read())

def convertToImage(sbase64):
    return base64.b64decode(sbase64)

def createJSON(fname):
    #Apro il file JSON iniziale e lo converto in oggetto python
    with open("JSON/100_1000_12112019_19_keypoints.json", "r") as json_file:
        j =  json.loads(json_file.read())

    fo

    #Aggiungo il campo image
    print(j["version"])
    j["image"] = "pippo"
    print(j["image"])

    #Salvo il JSON finale
    with open("RESULT/" + fname, "w") as json_result:
        json_result.write(json.dumps(j))

for file in os.listdir("JSON/"):
    createJSON(file)

#sbase64 = convertoToBase64("test.png")
#result_file = open("result.png", "wb")
#result_file.write(convertToImage(sbase64))
#result_file.close()