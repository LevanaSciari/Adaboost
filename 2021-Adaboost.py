#!/usr/bin/env python
# coding: utf-8

from itertools import combinations
from numpy import log as ln
from numpy import exp as e
import random

#class Point
class point:

#init point
    def __init__(self, x=0, y=0, label=0):
        self.x = float(x)#coordinate x
        self.y = float(y)#coordinate y
        self.label = int(label)#label of the point
        self.weight = 0#weight of the point

#class Line
class  Line:

#init line with two points
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
        if point1.x - point2.x == 0:
            self.parallelY = True
            self.m = 0#it is the slope of the line
            self.n = 0
        else:
            self.parallelY = False
            self.m = (point1.y - point2.y) / (point1.x - point2.x)#it is the slope of the line
            self.n = point1.y - (self.m * point1.x)  #find the n int the equation y=mx+n



#this function receive point and the line
#the line predict the label of the point
    def predictLabel(self, point_other):
        # if the rule is not a line parallel to the Y axis
        if not self.parallelY:
            y_line = (self.m * point_other.x) + self.n
            if point_other.y >= y_line:
                return -1
            else:
                return 1
        # if the rule is a line parallel to the Y axis
        else:
            if point_other.x >= self.point1.x:
                return 1
            else:
                return -1        

#this function return the best line .
#the best line is the line with the minimum error on the classify point
def Best_Line(points):
    Combination=combinations(points, 2)
    #we do all combination between two point
    #it it all possibilities of the rule(all possibilities of line between two points)
    #for two point we receive two rule
    best_line=Line(point(), point())
    min_sum_error = 1  # min sum of errors
    for i in Combination:
        new_line = Line(i[0], i[1])
        temp_sum = 0
        for x in points:
            #we check if the label of the point is same or note same that the label that the line predict
            #we do as we learned in class
            #we check the points for which the line predicted false
            classify = int(x.label != new_line.predictLabel(x))
#we add the weight of the point that the rule does not classify them good
            temp_sum += x.weight*classify
            # if label point is equal to the predict label
            # #we will get zero and mult the weight by 0.
            # if label point is not equal to the predict label (false)
            # we will mult in 1 and get the weight to calculate the error.
        if temp_sum< min_sum_error and temp_sum!=0:
            min_sum_error=temp_sum
            best_line= new_line
    return(best_line,min_sum_error)#return best line and his error

#get dat from the file rectangle.txt
    
def get_data_for_rectangle(file_name):
    file = open(file_name, "r")
    data = []
    for line in file:
        new_line = line
        str_points = new_line.split()
        line=point(str_points[0], str_points[1], str_points[2])
        data.append(line)
    return data
    
#modify the weight of the points
def Edit_Weight_Adaboost(points, r):
    bestLines =[]
    for x in points:
        x.weight= 1/len(points)
    for t in range(r):
        bestLine,errorLine=Best_Line(points)
        if errorLine>0.5:
            print("error too big:",errorLine)
            break
        alpha_t = 0.5 * ln((1 - errorLine) / errorLine)
        Z =0
        for x in points:
            x.weight= x.weight * e(-alpha_t * bestLine.predictLabel(x) * x.label)
            Z+=x.weight
        for x in points:
            x.weight=x.weight/Z
        bestLines.append((bestLine,alpha_t))
    return bestLines

#algorithm adaboost As in the presentation of the lesson
def run_adaboost(adaboost_rounds, rounds, points_set):
    s = "Adaboost:"
    for r in range(1,adaboost_rounds):
        testing_total =0
        training_total =0
        for i in range(rounds):
            random.shuffle(points_set)
            half_data = int(len(points_set) / 2)
            Training= points_set[0:half_data]
            Testing=points_set[half_data:]
            Result= Edit_Weight_Adaboost(Training , r)
            for x in Testing:
                H=0
                for best_line, error_best_line in Result:
                    H+=error_best_line*best_line.predictLabel(x)
                testing_total +=int(H*x.label<0)
            for x in Training:
                H=0
                for best_line, error_best_line in Result:
                    H += error_best_line*best_line.predictLabel(x)
                training_total+=int(H*x.label<0)
        Testing_errors=(testing_total/rounds)/half_data
        Training_errors=(training_total/rounds)/half_data
        Test_error="\nThe error  Hk on the test set " +str(r)+": "+ "%.3f" % Testing_errors+"(" +"%.3f"%(1-Testing_errors)+" % were correct)"
        Training_error="\nThe error  Hk on training set " +str(r)+": "+"%.3f" % Training_errors+"(" +"%.3f"%(1-Training_errors)+" % were correct)"
        s+=Training_error
        s+=Test_error
        print(Training_error)
        print(Test_error)
        s+="\n"
    return s


           
#--------main----------
if __name__ == '__main__':

    data_set_rectangle=get_data_for_rectangle("rectangle.txt")
    file=open("Output.txt", "w+")
    result=run_adaboost(adaboost_rounds=9, rounds=100, points_set=data_set_rectangle )
    file.write(result)
    file.close()

