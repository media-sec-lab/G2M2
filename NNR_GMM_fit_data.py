

import tensorflow as tf
import numpy as np


##### 
##### num is the number of Gaussians.
##### dim is the dimension of data.
##### Nz=Nz1*Nz2
num=100
dim=2
BATCH_SIZE = 1
Nz1=1024
Nz2=1024
Nx=60000
NUM_ITER =20000
UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'
LEARNING_RATE =0.1
LEARNING_RATE_DECAY = 0.9
MOMENTUM = 0.9
decay_step = 20
is_train = True
data_path = 'gm/'
data_name='Y100_'


x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, Nx, 1, dim])
z = tf.placeholder(tf.float32,shape=[BATCH_SIZE, Nz1, Nz2, dim])



#Parameter initialization
Standard_Deviation=0.6
b1_init=np.zeros([1,1,dim,num],dtype=np.float32)
b2_init=np.zeros([1,1,dim,num],dtype=np.float32)

b1_init[:,:,0,:]=1/(Standard_Deviation)
b2_init[:,:,1,:]=1/(Standard_Deviation)

t1_init=np.zeros([num],np.float32)
t2_init=np.zeros([num],np.float32)
    
lamda_init=np.ones([1,1,num,1],dtype=np.float32)/num






np.set_printoptions(threshold=100000)
is_train = tf.placeholder(tf.bool,name='is_train')





#####Network structure
with tf.variable_scope("Group1") as scope:  
    b1=tf.Variable( b1_init,name="b1" )
    b2=tf.Variable( b2_init,name="b2" )
    
    t1=tf.Variable(t1_init,dtype=tf.float32,name="t1")
    t2=tf.Variable(t2_init,dtype=tf.float32,name="t2")

    
    lamda=tf.Variable( lamda_init,name="lamda" )
    
    conv1_1=tf.nn.conv2d(x, b1, [1,1,1,1], padding='SAME',name="conv1_1"  )
    conv1_2=tf.nn.conv2d(x, b2, [1,1,1,1], padding='SAME',name="conv1_2"  )
 
    
    
    Y1_1= tf.nn.bias_add(conv1_1,t1) 
    Y1_2= tf.nn.bias_add(conv1_2,t2) 


    
    conv_Y1_1=tf.square(Y1_1)
    conv_Y1_2=tf.square(Y1_2)

    exp_input=conv_Y1_1+conv_Y1_2
    exp_input=tf.multiply(-0.5,exp_input)
    exp_result=tf.exp(exp_input)
    
    
    p_x=tf.nn.conv2d(exp_result,lamda, [1,1,1,1], padding='VALID',name="p_x"  )
    
   
    
    I_pq=tf.reduce_mean(p_x,[1,2])
    S_alpha=tf.square(I_pq)
    
with tf.variable_scope('Group2') as scope:
    conv2_1=tf.nn.conv2d(z, b1, [1,1,1,1], padding='SAME',name="conv2_1"  )
    conv2_2=tf.nn.conv2d(z, b2, [1,1,1,1], padding='SAME',name="conv2_2"  )
  

    
    Y2_1= tf.nn.bias_add(conv2_1,t1) 
    Y2_2= tf.nn.bias_add(conv2_2,t2) 
 

    conv_Y2_1=tf.square(Y2_1)
    conv_Y2_2=tf.square(Y2_2)
 
 
    

    
    exp_input=conv_Y2_1+conv_Y2_2
    exp_input=tf.multiply(-0.5,exp_input)
    exp_result=tf.exp(exp_input)
    
    
    p_z=tf.nn.conv2d(exp_result,lamda, [1,1,1,1], padding='VALID',name="p_z"  )
    p_z_square=tf.square(p_z) 
    S_beta=tf.reduce_mean(p_z_square,[1,2])
    C_z=tf.reduce_mean(p_z,[1,2])




##### loss function l_BUSCS 
with tf.variable_scope('Group1') as scope:
    
    min_function=S_beta/S_alpha
    

vars = tf.trainable_variables()
params = [v for v in vars if ( v.name.startswith('Group1/') or v.name.startswith('Group2/')) ]


loss =min_function
tf.summary.histogram("loss", loss)
global_step = tf.Variable(0,trainable = False)
decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, decay_step, LEARNING_RATE_DECAY, staircase=True)
opt = tf.train.AdamOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params)

variables_averages = tf.train.ExponentialMovingAverage(0.95)
variables_averages_op = variables_averages.apply(tf.trainable_variables())



data_x = np.zeros([BATCH_SIZE,Nx,1,dim])
test_x = np.zeros([BATCH_SIZE,Nx,1,dim])
data_z = np.zeros([BATCH_SIZE,Nz1,Nz2,dim])


saver = tf.train.Saver()
merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
     writer = tf.summary.FileWriter("logs/logs_aug_1/train/shi_aug_s0.4_0731",sess.graph)
     tf.global_variables_initializer().run()
     summary = tf.Summary()
     noise=np.random.normal(0,0.2,600)	
     noise1=3*noise-4
     noise2=noise-2
     data_x[0,:,0,0]=np.genfromtxt(data_path+data_name+'1.txt', delimiter=',', dtype=np.float32)
     data_x[0,:,0,1]=np.genfromtxt(data_path+data_name+'2.txt', delimiter=',', dtype=np.float32)
     test_x=data_x
     data_x[0,0:600,0,0]=noise1
     data_x[0,0:600,0,1]=noise2

     

    

     for i in range(1,NUM_ITER+1):
         data_z[0,:,:,:] =np.random.rand(Nz1,Nz2,dim)*8-4
         _,l = sess.run([opt,loss],feed_dict={x:data_x,z:data_z,is_train:True})
         
         
         if i%100==0:  
             summary.ParseFromString(sess.run(merged,feed_dict={x:data_x,z:data_z,is_train:True}))
             writer.add_summary(summary, i)
         if i%100==0:  
             print ('batch result')
             print ('epoch:', i)
             print ('loss:', l)
             print (' ')
         if i%(10000)==0:
            saver.save(sess,'model'+str(i)+'.ckpt') 
            
         if i%(20)==0:
           
            C = sess.run(C_z,feed_dict={x:test_x,z:data_z})*8*8
            
            
#####       Save the normalized parameters of lamda            
            k1=sess.run(b1,feed_dict={x:data_x,z:data_z})
            k2=sess.run(b2,feed_dict={x:data_x,z:data_z})
            bias1=sess.run(t1,feed_dict={x:data_x,z:data_z})
            bias2=sess.run(t2,feed_dict={x:data_x,z:data_z})
            kd=sess.run(lamda,feed_dict={x:data_x,z:data_z})
            np.savetxt('k1'+data_name+str(num)+'.txt',np.squeeze(k1))
            np.savetxt('k2'+data_name+str(num)+'.txt',np.squeeze(k2))
            np.savetxt('bias1'+data_name+str(num)+'.txt',np.squeeze(bias1))
            np.savetxt('bias2'+data_name+str(num)+'.txt',np.squeeze(bias2))
            np.savetxt('kd'+data_name+str(num)+'.txt',np.squeeze(kd)/C)
         
