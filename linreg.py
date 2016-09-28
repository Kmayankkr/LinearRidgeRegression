
#!/usr/bin/python    

import sys
import numpy as np
import scipy
import cython
import random
import copy 
import matplotlib.pyplot as plt

# 0 to 4176  and  0 to 11

rows = 4177  
cols = 12 
attr = 11

datacopy = []
dataset = np.zeros( (rows , cols) ) 
datacounter = 0

train = [] 
test = [] 

X = []
Y = []
I = [] 
W = []

TX = []
TY = []

RES = []
TRES = [] 

MSR = 0.0 


def read_input() :

	global datacopy, dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES 

	inputfile = open("linregdata", 'r')

	temp = inputfile.readline()	

	while(temp) : 
	
		array = temp.split(',') 

		# F, I, M 

		dataset[datacounter][0] = 1.0
		dataset[datacounter][1] = 0.0
		dataset[datacounter][2] = 0.0
		dataset[datacounter][3] = 0.0

		if   array[0] == 'M' :
			dataset[datacounter][3] = 1.0
		elif array[0] == 'I' :
			dataset[datacounter][2] = 1.0
		elif array[0] == 'F' :
			dataset[datacounter][1] = 1.0 
		
		for i in range(4, cols) :
			dataset[datacounter][i] = float(array[i-3]) 
			
		datacounter+= 1 
		
		temp = inputfile.readline()    
	
	datacopy = copy.deepcopy(dataset)
		
	
		
		
			
def standardize() : 
			
	global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES

	for i in range(1, cols-1) : 
		
		mean = 0.0
		stdev = 0.0
		
		for j in range(0, datacounter) : 
			mean+= dataset[j][i] 
		
		mean/= datacounter 
		 
		for j in range(0, datacounter) :
			stdev+= (dataset[j][i] - mean) ** 2 
			
		stdev/= datacounter 
		stdev = stdev ** 0.5 
		
		
		for j in range(0, datacounter) :
			dataset[j][i] = (dataset[j][i] - mean) / stdev 
			
		#print mean,
		#print " ",
		#print stdev 
		
	
		

		
		
def stancheck() : 
			
	global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES

	for i in range(0, cols-1) : 
		
		mean = 0.0
		stdev = 0.0 
		
		for j in range(0, datacounter) : 
			mean+= dataset[j][i] 
			
		mean/= datacounter 
		 
		for j in range(0, datacounter) :
			stdev+= (dataset[j][i] - mean) ** 2 
			
		stdev/= datacounter 
		stdev = stdev ** 0.5 
		
		#print mean,
		#print " ",
		#print stdev 
		
	
def partition(frac) :     # train / test

	 global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES
	 
	 np.random.shuffle(dataset)  
	 
	 nos = int(frac * rows)
	 
	 X = dataset[0:nos,0:attr]
	 TX = dataset[nos:rows,0:attr]
	 
	 Y = dataset[0:nos,attr:]
	 TY = dataset[nos:rows,attr:]
	 
	 I = np.eye( attr ) 
	 


def true_standardize() : 
			
	global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES

	nos = X.shape[0] 
	
	TTX = np.transpose( X )

	for i in range(1, attr) : 
		
		
		mean = np.mean(TTX[i])
		stdev = np.std(TTX[i]) 
		
		for j in range(0, nos) :
			X[j][i] = (X[j][i] - mean) / stdev 	
			
		for j in range(0, rows-nos) :
			TX[j][i] = (TX[j][i] - mean) / stdev
			
			
			
			

def mylinridgereg(lamda) :

	global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES
	
	W = np.dot ( np.transpose( X ) , X )
	
    #I = lamda * I
	
	W = W + I
	
	W = np.linalg.inv(W)
	
	W = np.dot ( W, np.transpose( X ) )
	
	W = np.dot ( W, Y )
	
	#print W    
	


def mylinridgeregeval() : 

	global dataset, datacounter, train, test, rows, cols, attr, X, Y, I, W, TX, TY, RES, TRES

	RES = np.dot ( X, W )
	TRES = np.dot ( TX, W )



def meansquarederr(A, B) :

	global MSR   

	MSR = 0.0  

	length = A.shape[0]    
	
	for i in range(length) :
		MSR+= (A[i][0] - B[i][0]) ** 2
		
	MSR/= length 
	
	#print MSR 

	


read_input()



fraction = [0.50, 0.60, 0.70, 0.80, 0.90]
lamd = [0.01, 0.1, 1, 5, 10, 20, 50, 70, 100]
train_error = [[0.0 for x in range(9)] for y in range(5)] 
test_error = [[0.0 for x in range(9)] for y in range(5)] 

train_min =  [0.0 for y in range(5)]
test_min =  [0.0 for y in range(5)]

dataset = copy.deepcopy(datacopy) 

trfrac = input("Enter Training set fraction : ")
penal = input("Enter Lambda (Penalty Parameter) : ")

print
print "Training set fraction : ",
print trfrac
print "Lambda (Penalty Parameter) : ",
print penal 

partition(trfrac)

true_standardize()

mylinridgereg(penal)

mylinridgeregeval()

meansquarederr(Y, RES)

print "Training mean square error : ",
print MSR

meansquarederr(TY, TRES)

print "Test mean square error : ",
print MSR



