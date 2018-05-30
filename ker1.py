from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
import numpy as np
seed =7
np.random.seed(seed)
dataframe=pd.read_csv("BBCN.csv")
array=dataframe.values
x=array[:,0:11]
y=array[:,11]

model=Sequential()
model.add(Dense(11,input_dim=11,bias_initializer='uniform',activation='relu'))
model.add(Dense(8,bias_initializer='uniform',activation='relu'))
model.add(Dense(8,bias_initializer='uniform',activation='relu'))
model.add(Dense(8,bias_initializer='uniform',activation='relu'))
model.add(Dense(1,bias_initializer='uniform',activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=375,batch_size=10)

scores=model.evaluate(x,y)

print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x,y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))