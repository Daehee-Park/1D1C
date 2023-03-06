# 딥러닝 쌩기초, Linear Regression

import tensorflow as tf

# define data
height=[170, 180, 160, 190, 150]
shoes=[260, 280, 240, 300, 200]

# define weight
a=tf.Variable(0.1)
b=tf.Variable(0.2)

#define loss function
def loss():
    y_pred=a*height+b
    return tf.reduce_mean(tf.square(y_pred-shoes))

#define optimizer
optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)
optimizer.minimize(loss, var_list=[a,b])

for i in range(300):
    optimizer.minimize(loss, var_list=[a,b])
    if i%10==0:
        print("a={}, b={}, loss={}".format(a.numpy(), b.numpy(), loss().numpy()))

#predict
input=180
y_pred=a*input+b
print("input={}, y_pred={}".format(input, y_pred.numpy()))