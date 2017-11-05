#coding=utf-8

import tensorflow as tf

#基本测试示例
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result = a+b
sess = tf.Session()
sum = sess.run(result)
sess.close()
print(sum)

#计算图g1
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",initializer=tf.zeros_initializer(),shape=[2,3])

#计算图g2
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",initializer=tf.ones_initializer(),shape=[2,3])

#在计算图g1中读取变量v
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))


#在计算图g2中读取变量v
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

#指定设备
g = tf.Graph()
with g.device('/gpu:0'):
    result = a+b
    sess = tf.Session()
    sum = sess.run(result)
    sess.close()
    print(sum)


#简单的前向传播
import tensorflow as tf 

w1 = tf.Variable(tf.random_normal([2,3],stddev=1.0,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1.0,seed=1))

#input_layer = tf.constant([[0.7,0.9]])
input_layer = tf.placeholder(tf.float32,shape=(1,2),name="input_layer")

print(input_layer.get_shape())
print(tf.transpose(input_layer).get_shape())

a=tf.matmul(input_layer,w1)
y=tf.matmul(w1,w2)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#print(sess.run(y))
print(sess.run(y,feed_dict={input_layer:[[0.7,0.9]]}))

sess.close()


#简单的反向传播的训练
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1.0,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1.0,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

##生成用于训练的模拟数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_opt = tf.initialize_all_variables()
    sess.run(init_opt)
    
    print(sess.run(w1))
    print(sess.run(w2))


    #训练轮数
    STEPS= 5000
    for i in range(STEPS):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if(i%500 == 0):
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("after %d tranning steps, cross entropy on all data is %g" % (i,total_cross_entropy))
    

    print(sess.run(w1))
    print(sess.run(w2))