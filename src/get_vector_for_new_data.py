import numpy as np 

def get_vector(comment):

	dic = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

	cmt = list(comment)
	vector = [dic.index(i)+1 for i in cmt if i in dic]

	max_length=2401

	if len(cmt)<=max_length:
		vector_input = np.array([vector + [0]*(max_length-len(vector))])
	else:
		vector_input = vector[0:max_length]

	vector_input = vector_input.reshape(1, max_length)

	return vector_input



