
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)
    
class MarvelllousKNN():
    def fit(self,TrainingData,Trainingtarget):
        self.TrainingData=TrainingData
        self.Trainingtarget=Trainingtarget
        
    def predict(self,TestData):
        prediction=[]
        
        for row in TestData:
            lebel=self.closest(row)
            prediction.append(lebel)
        return prediction
        
    def closest(self,row):
        bestdistance=euc(row,self.TrainingData[0])
        bestindex=0
        for i in range(1,len(self.TrainingData)):
            dist=euc(row,self.TrainingData[i])
            if dist<bestdistance:
                bestdistance=dist
                bestindex=i
            
        return self.Trainingtarget[bestindex]
    
def KNeighborsClassifierAccuracy():
    iris=load_iris()
    
    target=iris.target
    Data=iris.data
    
    data_train, data_test,target_train, target_test=train_test_split(Data,target,test_size=0.5)
    
    classifier=MarvelllousKNN()
    
    classifier.fit(data_train,target_train)
    
    predict=classifier.predict(data_test)
    
    Accuracy=accuracy_score(target_test,predict)
    
    return Accuracy

def main():
    
    Accuracy=KNeighborsClassifierAccuracy()
    print("KNeighborsClassifierAccuracy:",Accuracy*100,"%d")
    
    
if __name__=="__main__":
    main()