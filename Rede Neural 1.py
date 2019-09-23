import pandas as pd
import numpy as np
import re, os, sys
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

lmtz = WordNetLemmatizer()
stoplist = stopwords.words('english')
lmtzlist = [1]*len(stoplist)
jArray = [1]*len(stoplist)
count = 0

j = 0
while j <= len(stoplist)-1:
    lmtzlist[j] = lmtz.lemmatize(stoplist[j])
    j = j+1

    
j = 0
while j <= len(stoplist)-1:
    if lmtzlist[j] != stoplist[j]:
        jArray[count] = j
        count = count + 1
    
    j = j+1
    
count = count + len(stoplist)
newStopList = [1]*count

j = 0
while j<= count-1:
    if j <= len(stoplist)-1:
        newStopList[j] = stoplist[j]
    else:
        newStopList[j] = lmtzlist[jArray[count-len(stoplist)]]
    j = j+1
    
df = pd.read_csv("data/Train_rev1.csv")
df.drop(columns=['Id','LocationRaw','ContractType','SalaryRaw','SourceName'],axis=1,inplace=True)
df['Title'].fillna('untituled', inplace=True)
df['FullDescription'].fillna('undrecribe', inplace=True)
df['LocationNormalized'].fillna('unlocate', inplace=True)
df['ContractTime'].fillna('doubt', inplace=True)
df['Company'].fillna('unknown', inplace=True)
df['Category'].fillna('unspecified', inplace=True)
name_dict = {0 :'Title', 1 :'FullDescription', 2 :'LocationNormalized',  3 :'Category', 4:'ContractTime', 5:'Company' }
df.head()

def textos(x):
    texto = [None]*len(df[str(name_dict[x])])
    i = 0
    while i < len(df[str(name_dict[x])]):
        texto[i] = str(df[str(name_dict[x])][i])
        i += 1
    return texto

def pre_procesar(x):
    df_procesado = {}
    counter = 0
    for s in textos(x):
        s = s.lower()
        s = re.sub(r'[^\w]', ' ', s)
        s = re.sub(r'\b[a-z]\b', ' ', s)
        s = re.sub(r'\b[a-z][a-z]\b', ' ', s)
        s = re.sub(r'\b[0-9]\b', ' ',  s)
        s = re.sub(r'\b[0-9][0-9]\b', ' ', s)
        s = re.sub(r'\b[0-9][0-9][0-9]\b', ' ', s)
        s = re.sub(r'\b[0-9][0-9][0-9][0-9]\b', ' ', s)
        s = re.sub(r'\b[0-9][0-9][0-9][0-9][0-9]\b', ' ', s)
        s = re.sub(r'\b[0-9][0-9][0-9][0-9][0-9][0-9]\b', ' ', s)
        s = re.sub(r'[^\w.]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.split(',')
        df_procesado[counter] = s
        counter += 1
    return df_procesado 

def procesado(x):
    texto_procesado = str([ v for v in pre_procesar(x).values() ])
    texto_procesado = texto_procesado.replace('[[','[')
    texto_procesado = texto_procesado.replace(']]',']')
    texto_procesado = texto_procesado.replace('[','')
    texto_procesado = texto_procesado.replace("'",'')
    texto_procesado = texto_procesado.replace(",",'')
    return texto_procesado

def convert_to_list(x):
    string = procesado(x)
    convertido = string.split(']')
    return convertido

def conver_to_one_string():
    temp5 = convert_to_list(5)
    temp4 = convert_to_list(4)
    temp3 = convert_to_list(3)
    temp2 = convert_to_list(2)
    temp1 = convert_to_list(1)
    temp0 = convert_to_list(0)
    x = 0
    while x<len(temp0):
        temp0[x] = temp0[x]+' '+temp1[x]+' '+temp2[x]+' '+temp3[x]+' '+temp4[x]+' '+temp5[x]
        x+=1
    convert_list = temp0
    return convert_list  

def fazer_dicionario(x):
    dicionario = procesado(x)
    dicionario =  dicionario.replace(']','')
    dicionario = dicionario.split(' ')
    return dicionario

def junta_dicionario(x):
    k = x-1
    dic = fazer_dicionario(x)
    while k >= 0:
        dic = dic + fazer_dicionario(k)
        k = k-1
    return dic

y_dataframe = df['SalaryNormalized'].values
x_dataframe = df[['Title','FullDescription','LocationNormalized','Category', 'ContractTime', 'Company' ]]

df_procesado = conver_to_one_string()

k = len(df_procesado)
x_train = df_procesado[:int(0.7*k)] 
x_val = df_procesado[int(0.70*k):int(0.85*k)]
x_test = df_procesado[int(k*0.85)::]

y_train = y_dataframe[:int(0.7*k)] 
y_val = y_dataframe[int(0.70*k):int(0.85*k)]
y_test = y_dataframe[int(k*0.85)::]

todas_palavras = junta_dicionario(5)

todas_palavras = set(todas_palavras)
todas_palavras = list(todas_palavras)

word_index = dict(enumerate(todas_palavras))
j=0
for frase in x_train:
    seq=frase.split()
    for term in seq:
        if term not in word_index.keys():
            word_index[term]=j
            j+=1
            
embeddings_index = {}
f = open(os.path.join('data/glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Se encontraron %s terminos con sus vectores de embedding.' % len(embeddings_index))


embedding_vector = 100
embedding_matrix = np.zeros((len(word_index.keys()), embedding_vector))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  
        
from keras.preprocessing import sequence
x_new_train = [[word_index[word] for word in text.split()] for text in x_train]
x_new_val = [[word_index[word] for word in text.split() if word in word_index] for text in x_val]

max_input_lenght = 1000
Xtrain = sequence.pad_sequences(x_new_train,maxlen=max_input_lenght,padding='post',value=0)
Xval = sequence.pad_sequences(x_new_val,maxlen=max_input_lenght,padding='post',value=0)

from keras.layers import Embedding , Flatten , Input , Dense , BatchNormalization , Activation, Dropout
from keras.models import Model
from keras import optimizers
from keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import mean_absolute_error

poo1 = 25
poo2 = 6
poo3 = 2
poo4 = 1
stride1 = 4
stride2 = 2
stride3 = 1
stride4 = 1
con = 128
con2 = 7

epo = 11

embedding_vector = 100

embedding_layer = Embedding(input_dim=len(word_index.keys()),output_dim=embedding_vector,weights=[embedding_matrix],
                        input_length=max_input_lenght,trainable=False)
sequence_input = Input(shape=(max_input_lenght,))
embedded_sequences = embedding_layer(sequence_input)

conv1 = Conv1D(con, con2, activation='relu',padding='same')(embedded_sequences)
pool1 = MaxPooling1D(pool_size=poo1, stride = (stride1))(conv1)
conv2 = Conv1D(con, con2, activation='relu',padding='same')(pool1)
pool2 = MaxPooling1D(pool_size=poo2, stride=(stride2))(conv2)
conv3 = Conv1D(con, con2, activation='relu',padding='same')(pool2)
pool4 = MaxPooling1D(pool_size=poo3, stride=(stride3))(conv3)
conv4 = Conv1D(con, con2, activation='relu',padding='same')(pool4)
pool5 = MaxPooling1D(pool_size=poo4, stride=(stride4))(conv4)
flat = Flatten()(pool5)

preds = Dense(1, activation='linear')(flat)
model = Model(sequence_input, preds)
model.summary()

model.compile(loss='mse',optimizer='RMSprop',metrics=['acc'])
model.fit(Xtrain, y_train, validation_data=(Xval, y_val),epochs=epo, batch_size=256)

print("MAE on train: ",mean_absolute_error(y_train, model.predict(Xtrain)))
print("MAE on validation: ",mean_absolute_error(y_val, model.predict(Xval)))