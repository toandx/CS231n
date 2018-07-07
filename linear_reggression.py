import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras import optimizers
X=np.random.rand(100,2)
y = 100*X[:,0]+300*X[:,1]+np.random.randn(100)*1
X_test=np.random.rand(20,2)
y_test=3*X_test[:,0]+2*X_test[:,1]+np.random.randn(20)*0.01
sgd = optimizers.SGD(lr=0.1) # important
model=Sequential()
model.add(Dense(1,input_shape=(2,),activation='linear'))
model.compile(loss='mse',optimizer=sgd)
model.fit(X,y,epochs=100,verbose=1,batch_size=1,validation_data=(X_test,y_test))
print(model.get_weights())
