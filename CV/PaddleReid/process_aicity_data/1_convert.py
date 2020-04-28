# convert to utf8
with open('train_label.xml', 'r') as myfile:
    dstr = myfile.read()
    #dstr = dstr.decode('gb2312').encode('utf-8')
    dstr = dstr.replace('gb2312', 'utf-8')
    text_file = open("train_label_utf8.xml", "w")
    text_file.write(dstr)
    text_file.close()
