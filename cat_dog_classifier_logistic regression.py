import numpy as np
import matplotlib.pyplot as pt;
import scipy
import h5py
from scipy import  ndimage
from PIL import Image
import glob                                                                                                             
import cv2
img_dim = 160

# This function loads the images from the folder 
def load_images():
    images = glob.glob("train\\*.jpg")
    np.random.shuffle(images)
    
    images= images[:5]
    
    
    x = np.zeros((1,img_dim*img_dim*3))
    y = np.zeros((len(images),1))
    cnt=0
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, (img_dim,img_dim))
        img = np.asarray(img)
        
        cv2.imshow("first", img)
        
        img = np.reshape(img,(1,-1))
        x = np.append(x,img,axis = 0)
        is_cat = image.split('.')[0]=='train\\cat'
        y[cnt][0] =1 if  is_cat else 0
        if cnt%100==0:
            print("loaded ",cnt,"images")
        cnt+=1
    cv2.waitKey(0)
    print(images[0])
    x = np.delete(x, 0,axis = 0)
    print(x.shape)
    for i in range (len(x)):
        img = x[i]
        img = np.reshape(img, (img_dim, img_dim, 3))
        cv2.imshow("first"+str(i), img)
        cv2.waitKey(0)
    x = x.astype(int)
    x = x/255
    print(x.shape, y.shape)
    return x,y

# supporting function to display images
def display_image(x):
    for i in range(x.shape[0]):
        img = np.reshape(x[i],(img_dim,img_dim,3))
        print(img)
        img = Image.fromarray(img,'RGB')
        img.show()




# split the data in required format that is train and test dataset.
def split_data(x, y, split_ratio = 0.8):
    split_point = (int)(x.shape[0]*split_ratio)
    x_train = x[:split_point]
    y_train = y[:split_point]
    x_test = x[split_point:]
    y_test = y[split_point:]
    return x_train, y_train, x_test, y_test

def sigmoid(x):
    res = 1/(1+np.exp(np.multiply(-1,x)))
    return res


def initialize_weights():
    return np.zeros(shape = (img_dim*img_dim*3,1)),np.zeros(shape=(1))

# calculates the end values of the input according  to the current weights and returns to the predict function.
def propagate(w,b,x,y):
    m = x.shape[0]

    predict = np.matmul(x,w)+b
    
    predict = sigmoid(predict)
    cost = (-1/m)*(np.sum(np.multiply(y,np.log(predict))+np.multiply((1-y),np.log(1-predict))))
    dw = np.reshape((1/m)*np.sum((predict-y)*x,axis = 0),(-1,1))
    db = (1/m) *np.sum(predict-y)
    print
    grad = { "dw": dw,
             "db": db}
    return grad, cost


# This functions optimizes the values of cost variable which brings the parameters to predict accuarately.
def optimize(w,b,x,y,num_iterations, learning_rate, print_cost = True):
    costs = []
    m = x.shape[0]
    assert(w.shape == (img_dim*img_dim*3,1))
    print(num_iterations)
    for i in  range(num_iterations):
      
        grad, cost = propagate(w,b,x,y)
        w = w - learning_rate*(grad["dw"])
        b = b - learning_rate*(grad["db"])

        if(True):
            costs.append(cost)
            if print_cost:
                print("Iteration no: ",i," cost: ",cost)
        params = {
            "w":w,
            "b":b}
    return params , cost


def predict(w,b,x):

    m = x.shape[0]
    y_predict = np.zeros( (1,m))
    w = np.reshape(w,(x.shape[1],1))
    y_predict = np.matmul(x,w)+b
     
    y_predict = sigmoid(y_predict)

    for i in range(m):
         y_predict[i,0] = 1 if y_predict[i,0]>0.5 else 0


    assert(y_predict.shape == (m,1))
    
    return y_predict

     
def get_accuracy(y,y_predict):
    m = y.shape[0]
    count_correct = 0
    wrong_prediction = []
    print(y.shape,y_predict.shape)
    for i in range (m):
        if(y[i][0] == y_predict[i][0]):
            count_correct+= 1
        else:
            wrong_prediction.append(i)

    return count_correct*100/m,wrong_prediction

    
def model_train(x_train, y_train, x_test, y_test, learning_rate = 0.2,num_iterations = 1000):
    
    m = x_train.shape[0]
    w,b = initialize_weights()
    print("in main")
    params, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate)
    w = params["w"]
    b = params["b"]
    y_test_predict = predict(w,b,x_test)
    y_train_predict = predict(w,b,x_train)
    accuarcy_train,wrong_prediction = get_accuracy(y_train, y_train_predict)
    print("accuracy: " ,accuarcy_train)
    accuracy_test,wrong_prediction = get_accuracy(y_test, y_test_predict)
    d = {"w": w,
         "b": b,
         "y_train_predict": y_train_predict,
         "accuarcy_train": accuarcy_train,
         "y_test_predict": y_test_predict,
         "accuracy_test": accuracy_test,
         "leanining_rate" : learning_rate,
         "num_iterations": num_iterations
         }
    return d



        

def main():
    print("lading images!!")
    x, y =load_images()
    
    x_train , y_train, x_test, y_test = split_data(x,y)
    d = model_train(x_train , y_train, x_test, y_test, 0.5,10)
    #print(d)

if __name__=="__main__":
   main()

s
