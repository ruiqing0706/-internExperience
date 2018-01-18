
# coding: utf-8

# # TensorFlow实现进阶的卷积神经网络

# In[1]:


import cifar10 # 将下载的TensorFlow Models库里的cifar10放到项目中
import cifar10_input # 将cifar文件夹中的cifar10_input.py复制到外层文件夹中
import tensorflow as tf
import numpy as np
import time
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


# 定义训练轮数max_steps
max_steps = 3000
# 定义batch_size
batch_size = 128
# 定义下载CIFAR-10数据的路径
data_dir = 'temp/cifar10_data/cifar-10-batches-bin'


# In[3]:


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


# In[4]:


# cifar10.maybe_download_and_extract() # 'module' object has no attribute 'maybe_download_and_extract'
# download_url: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 下载的数据需要添加 .bin 的后缀，才能对数据做处理。
# 准备训练数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# 准备测试数据
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)


# In[5]:


# 创建输入数据的placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])


# In[6]:


# 创建第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# In[7]:


# 创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[8]:


# 创建全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)


# In[9]:


# 类似全连接层，隐含节点数下降一半
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)


# In[10]:


# 最后一层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# In[11]:


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# In[12]:


loss = loss(logits, label_holder)


# In[13]:


# 选择Adam优化器，设定学习速率为1e-3
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


# In[14]:


# 输出Top k
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)


# In[15]:


# 创建session
session = tf.InteractiveSession()
# 变量全局初始化

tf.global_variables_initializer().run()


# In[16]:


# 启动图片数据增强线程队列
tf.train.start_queue_runners()


# In[ ]:


# 正式开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = session.run([images_train, labels_train])
    _, loss_value = session.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    print("...")
    if step % 1 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        print("...")
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
