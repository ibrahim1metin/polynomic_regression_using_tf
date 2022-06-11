import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
class poly():
    def __init__(self,poly_deg):
        self.poly_deg=poly_deg  
    def predict(self,x):
        return tf.transpose(tf.linalg.matmul(x,self.w,transpose_a=True,transpose_b=True))
    def train(self,x,y,lr,epochs,batch_size):
        val=[]
        self.batch_size=batch_size
        self.w=tf.Variable(tf.random.normal((1,self.poly_deg+1)))
        xval=tf.placeholder(tf.float32,(self.poly_deg+1,batch_size))
        yval=tf.placeholder(tf.float32,(1,batch_size))
        y_pred=self.predict(xval)
        cost=tf.math.reduce_mean(tf.math.square(y_pred-yval))
        opt=tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(cost)
        init=tf.compat.v1.global_variables_initializer()
        sess=tf.compat.v1.Session()
        sess.run(init)
        for i in range(epochs):
            ep_loss=[]
            for j in range(len(x)//batch_size):
                x1=x[j*batch_size:min(len(x),(j+1)*batch_size)]
                y1=[y[j*batch_size:min(len(y),(j+1)*batch_size)]]
                x1_r=[]
                for k in x1:
                    x1_r_s=[]
                    for r in range(self.poly_deg+1):
                        x1_r_s.append(k**r)
                    x1_r.append(x1_r_s)
                if i==range(epochs)[-1]:
                    a,_,c=sess.run((y_pred,opt,cost),feed_dict={xval:np.transpose(np.array(x1_r)),yval:y1})
                    val.append(a)
                    ep_loss.append(c)
                else:
                    _,c=sess.run((opt,cost),feed_dict={xval:np.transpose(np.array(x1_r)),yval:y1})
                    ep_loss.append(c)
            print(sum(ep_loss)/len(ep_loss))
        return val
x =  np.random.normal(0, 1, 2000)
y = 1.233*x - 2.9 * (x ** 2) + 0.56 * (x ** 3)+1.803
xd=x/max(x)
yd=y/max(y)
deg=3
poly_reg=poly(deg)
vals=poly_reg.train(x,y,0.0001,3000,100)
plt.scatter(xd, np.array(vals).reshape((2000,)),color="blue")
plt.plot(xd, y, 'ro')
plt.legend()
plt.show()
