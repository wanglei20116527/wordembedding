import random
import numpy as np
import tensorflow as tf 

a = tf.placeholder(dtype=tf.float32, shape=(None, 2, 4))
b = tf.placeholder(dtype=tf.float32, shape=(None, 2, 2))
c = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3))
e1 = tf.placeholder(dtype=tf.float32, shape=(None, 2))
e2 = tf.placeholder(dtype=tf.float32, shape=(None, 2))
d = tf.concat([a, b], 2)
e = tf.concat([e1, e2], 1)

asoftmax = tf.nn.softmax(a, 1)


with tf.Session() as session:
	tf.global_variables_initializer().run()

	# print(session.run(aPercent, feed_dict={
	# 	a: [
	# 		[
	# 			[1, 2, 3, 4],
	# 			[2, 3, 4, 5]
	# 		],
	# 		[
	# 			[1, 2, 2, 2],
	# 			[3, 3, 3, 3]
	# 		]
	# 	],
	# 	b: [
	# 		[
	# 			[8, 8],
	# 			[6, 6]
	# 		],
	# 		[
	# 			[10, 10],
	# 			[6, 6]
	# 		]
	# 	],
	# 	c: [
	# 		[
	# 			[5, 5, 5],
	# 			[6, 6, 6]
	# 		],
	# 		[
	# 			[7, 7, 7],
	# 			[8, 8, 8]
	# 		]
	# 	]
	# }))
	# a = session.run(tf.nn.softmax(e1), feed_dict={
	# 	e1: [
	# 		[1, 1],
	# 		[2, 5],
	# 		[3, 3]
	# 	],
	# 	e2: [
	# 		[4, 4],
	# 		[4, 4],
	# 		[4, 4]
	# 	]
	# })
	# print(type(a.tolist()))
	# print(a)	

	for i in range(10):
		for i in range(10):
			print(i)
		print('i:', i)













