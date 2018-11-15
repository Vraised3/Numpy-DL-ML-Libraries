## Refer: https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

import numpy as np

x_data = np.array([[0.1, 0.2, 0.7]])
y_label = np.array([[1.0, 0.0, 0.0]])

############## ACTIVATIONS###############
def Relu(in_data, deriv = False): # Works
    ReluOut = np.copy(in_data)
    if deriv == True:
        ReluOut[ReluOut > 0] = 1
    ReluOut[ReluOut < 0] = 0
    return ReluOut

def Sig(in_data, deriv = False): # Works
    sig_out = 1/(1+np.exp(-in_data))
    if deriv == True:
        return sig_out * (1 - sig_out)
    return sig_out

def SoftMax(in_data, deriv = False): # Works
    Exp = np.exp(in_data)
    Exp_sum = np.sum(Exp)
    out = Exp
    if deriv == True:
        for i in range(len(Exp)):
            out[i] = (Exp[i] * (Exp_sum - Exp[i])) / Exp_sum ** 2
        return out
    return Exp/Exp_sum

############### Error Function############

def CrossEnthropy(out, y_out, deriv = False): # Works
    n = out.size
    Error = -(1/n)*np.sum(y_out*np.log(out) + (1-y_out)*np.log(1 - out))
    if deriv == True:
        Error = -((y_out/out) - ((1 - y_out)/(1 - out)))
    return Error

##out = np.array([[0.198, 0.285, 0.516]])
##y_out = np.array([[1.0, 0.0, 0.0]])

def Layer(ip, weights, bias,deriv = False):
    z = np.dot(ip, weights) + bias
    if deriv == True:
        return ip
    return z

def MaxLhood(): # Maximum Likelyhood Classifier
    pass
############### Start Network############

##w1 = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.7], [0.4, 0.3, 0.9]])
##w2 = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.7], [0.6, 0.4, 0.8]])
##w3 = np.array([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.5, 0.2, 0.9]])
w1_size = 4
w2_size = 4
w3_size = 3
w1 = np.random.random((x_data.shape[1], w1_size))
w2 = np.random.random((w1.shape[1], w2_size))
w3 = np.random.random((w2.shape[1], w3_size))
alpha = 0.01 # Learning rate
bias1 = np.array([np.ones(w1.shape[1])])
bias2 = np.array([np.ones(w2.shape[1])])
bias3 = np.array([np.ones(w3.shape[1])])

epochs = 280
print_epochs = 30

for i in range(epochs):
    # Forward Prop
    z1 = Layer(x_data, w1, bias1)                                                 # Correct 1x3
    Hidden1 = Relu(z1)                                                              # Correct 1x3
    z2 = Layer(Hidden1, w2, bias2)                                              # Correct 1x3
    Hidden2 = Sig(z2)                                                                # Correct 1x3
    z3 = Layer(Hidden2, w3, bias3)                                              # Correct 1x3
    Output = SoftMax(z3) # Correct Webpage is Wrong             # Correct 1x3
    Error = CrossEnthropy(Output, y_label)                               # Correct 1
    ##    print(Error)

    # Backward Prop
    Error_Back = CrossEnthropy(Output, y_label, deriv = True) # # de/dOout      1x3
    Output_Back =  SoftMax(z3, deriv = True)                            # # dOout/dOin   1x3 i.e. Hidden3
    z3_Back = Layer(Hidden2, w3, bias3, deriv = True)               # # dOin/dWkl Hidden2
    Hidden2_Back = Sig(z2, deriv = True)                                 # # dh2out/dh2in 1x3
    z2_Back = Layer(Hidden1, w2, bias2, deriv = True)                  # dh2in/dWjk
    Hidden1_Back = Relu(z1, deriv = True)                                  # dh2in/dWjk
    z1_Back = Layer(x_data, w1, bias1, deriv = True)                     # dh1out/dh2in
    ##print(w2)

    del_Op = Error_Back * Output_Back
    ## Hidden Nodes
    del_H2 = Hidden2_Back * np.dot(w3, del_Op.T)
    del_H1 = Hidden1_Back * np.dot(w2, del_H2.T)

    w3_delta = del_Op * Hidden2
    w2_delta = del_H2 * Hidden1
    w1_delta = del_H1 * x_data
    
    w3 -= alpha * w3_delta
    bias3 -= alpha * w3_delta
    w2 = w2 - alpha * w2_delta
    bias2 -= alpha * w2_delta
    w1 = w1 - alpha * w1_delta
    bias1 -= alpha * w1_delta
    if(i%print_epochs == 0):
        print("Epoch : ", i, "  Error : ", Error)
