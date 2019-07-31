# -*- coding: utf-8 -*-
"""
@author: shaowu
baseline思路：用第1,2,..,n-1次的成绩，预测第n次的成绩；用第2,..,n次的成绩，预测第n+1次的成绩

这个只是其中的一种思路，希望对入门的有帮助，大佬的话可以忽视了
线上分数：好像8.26左右
"""
import pandas as pd
import numpy as np
from tqdm import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
def create_feature(data):
    '''
    提取特征：
    '''
    feats=[]
    for i,row in tqdm(data.iterrows()):
        m=[int(i) for i in row['history_score'] if int(i)>=50]
        feats.append([np.mean(m),np.median(m),np.std(m),np.max(m),np.min(m),\
                      np.mean(m[-2:]),np.std(m[-2:])
                      ])
        
    feats=pd.DataFrame(feats)
    feats.columns=['feats{}'.format(i) for i in range(feats.shape[1])]
    return feats
def xgb_model(new_train,y,new_test,lr,N):
  '''定义模型'''
  xgb_params = {'booster': 'gbtree',
          'eta':lr, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse',
          'silent': True,
          }
  #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
  skf=KFold(n_splits=N,shuffle=True,random_state=42)
  oof_xgb=np.zeros(new_train.shape[0])
  prediction_xgb=np.zeros(new_test.shape[0])
  for i,(tr,va) in enumerate(skf.split(new_train,y)):
    print('fold:',i+1,'training')
    dtrain = xgb.DMatrix(new_train[tr],y[tr])
    dvalid = xgb.DMatrix(new_train[va],y[va])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
    bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200, \
    verbose_eval=400, params=xgb_params)#,obj=custom_loss)
    oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
    prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
  print("stacking的score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_xgb, y))))
  prediction_xgb/=N
  return oof_xgb,prediction_xgb

#====================读入数据==========================================================
all_knowledge= pd.read_csv('初赛/train_s1/all_knowledge.csv')
course1_exams= pd.read_csv('初赛/train_s1/course1_exams.csv')
course2_exams= pd.read_csv('初赛/train_s1/course2_exams.csv')
course3_exams= pd.read_csv('初赛/train_s1/course3_exams.csv')
course4_exams= pd.read_csv('初赛/train_s1/course4_exams.csv')
course5_exams= pd.read_csv('初赛/train_s1/course5_exams.csv')
course6_exams= pd.read_csv('初赛/train_s1/course6_exams.csv')
course7_exams= pd.read_csv('初赛/train_s1/course7_exams.csv')
course8_exams= pd.read_csv('初赛/train_s1/course8_exams.csv')
exam_score= pd.read_csv('初赛/train_s1/exam_score.csv')
student= pd.read_csv('初赛/train_s1/student.csv')
course= pd.read_csv('初赛/train_s1/course.csv')
submission_s1=pd.read_csv('初赛/test_s1/submission_s1.csv')
#=======================================================================================
'''
简单地构造训练集：第1,2,...n-1次成绩去预测第n次的成绩；第2,...n次成绩去预测第n+1次的成绩；以此类推
'''
test_id=list(set(submission_s1['student_id']))
traindata=[]
for stu in tqdm(test_id): 
    history_grade=exam_score[exam_score.student_id==stu]
    student_test=submission_s1[submission_s1.student_id==stu]
    
    for i in range(1,9):
        m=history_grade[history_grade.course=='course'+str(i)]['score']
        traindata.append([stu,'course'+str(i),\
                      m.iloc[-1],\
                      list(m.iloc[:-1])
                      ])
traindata=pd.DataFrame(traindata,columns=['student_id','course','score','history_score'])
print('训练集构造完毕！')

'''
构造测试集：因为要预测未来两次的成绩，所以预测第n+2次时，是用到第n+1次的结果的
'''
testdata_one=traindata.copy()
testdata_one['history_score']=testdata_one['history_score'].apply(lambda x:x[1:])+\
                              testdata_one['score'].apply(lambda x:[x]) #从第2次成绩开始取,并把第n次的成绩加上
print('第一个测试集构造完毕！')

traindata=pd.concat([traindata,create_feature(traindata)],axis=1)
testdata_one=pd.concat([testdata_one,create_feature(testdata_one)],axis=1)
traindata=traindata.merge(student,how='left',on='student_id')
testdata_one=testdata_one.merge(student,how='left',on='student_id')

##第n次成绩的预测：
traindata=traindata[traindata.score>=50].reset_index(drop=True)
oof_xgb,prediction_xgb=\
xgb_model(np.array(traindata.drop(['student_id', 'course', 'score', 'history_score'],axis=1)),\
                         traindata['score'].values,\
                         np.array(testdata_one.drop(['student_id', 'course','score','history_score'],axis=1)),\
                         0.01,5)
testdata_one['score']=prediction_xgb

print('第二个测试集构造...')
testdata_two=testdata_one[['student_id', 'course', 'score', 'history_score']].copy()
testdata_two['history_score']=testdata_two['history_score'].apply(lambda x:x[1:])+\
                              testdata_one['score'].apply(lambda x:[x])
print('第二个测试集构造完毕！')

testdata_two=pd.concat([testdata_two,create_feature(testdata_two)],axis=1)
testdata_two=testdata_two.merge(student,how='left',on='student_id')

##第n+1次成绩的预测：                       
oof_xgb,prediction_xgb=\
xgb_model(np.array(traindata.drop(['student_id', 'course', 'score', 'history_score'],axis=1)),\
                         traindata['score'].values,\
                         np.array(testdata_two.drop(['student_id', 'course','score','history_score'],axis=1)),\
                         0.01,5)
testdata_two['score']=prediction_xgb
#====================================================================================
##准备提交数据：因为上面的两次预测不知道是属于哪一个exam_id的，所以做标记
#倒数第二次的课程exam_id：
exam_id1=[course1_exams['exam_id'].iloc[-2],course2_exams['exam_id'].iloc[-2],
          course3_exams['exam_id'].iloc[-2],course4_exams['exam_id'].iloc[-2],
          course5_exams['exam_id'].iloc[-2],course6_exams['exam_id'].iloc[-2],
          course7_exams['exam_id'].iloc[-2],course8_exams['exam_id'].iloc[-2]]
#倒数第一次的课程exam_id：
exam_id2=[course1_exams['exam_id'].iloc[-1],course2_exams['exam_id'].iloc[-1],
          course3_exams['exam_id'].iloc[-1],course4_exams['exam_id'].iloc[-1],
          course5_exams['exam_id'].iloc[-1],course6_exams['exam_id'].iloc[-1],
          course7_exams['exam_id'].iloc[-1],course8_exams['exam_id'].iloc[-1]]

##标记1和2，方便结合数据：
submission_s1['one']=submission_s1['exam_id'].apply(lambda x: 1 if x in exam_id1 else 2)
print(submission_s1['one'].value_counts())

testdata_one['one']=1 #第一次的预测结果标记为1
testdata_two['one']=2 #第二次的预测结果标记为2

result=submission_s1.copy()
result=result.merge(pd.concat([testdata_one,testdata_two],axis=0),\
                              how='left',on=['student_id','course','one'])

result=result[['student_id','course','exam_id','score']]
result.columns=['student_id','course','exam_id','pred']
result.to_csv('result.csv',index=None,encoding='utf8')