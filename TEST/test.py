import ijson

f = open('/mnt/c/Users/giuse/Desktop/DataSet.json', "r")
for item in ijson.items(f, "item"):
    print(item["id"])