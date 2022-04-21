function [outputArg1,outputArg2] = GNN(inputArg1,inputArg2)
clc ; clear;
% prepare data
X=load('group1.txt');
A=load('A_20_group1.txt');
Label=load('Label_Group1.txt');
ADLabels=load('Label_Group1.txt');
iteration=100;
LearningRate=0.0001;
[m,n]=size(X);
%fault object
Abnormal_number=60;
 
Layer2_hiddensize=2;
Layer3_hiddensize=m;
%Optionally, one can use the parameters we have trained
Layer2_w=load('Layer2_w.txt');
Layer2_b=load('Layer2_b.txt');
Layer3_w=load('Layer3_w.txt');
Layer3_b=load('Layer3_b.txt');
Layer2_e=load('Layer2_e.txt');
Layer3_e=load('Layer3_e.txt');
% or, one can use the random weight retrain.
% Layer2_w=rand(Layer2_hiddensize,m);
% Layer3_w=rand(Layer3_hiddensize,Layer2_hiddensize);
% Layer2_b=rand(Layer2_hiddensize,1);
% Layer3_b=rand(Layer3_hiddensize,1);
Layer2_output=rand(Layer2_hiddensize,1);
Layer3_output=rand(Layer3_hiddensize,1);
% Layer2_e=rand(Layer2_hiddensize,1);
% Layer3_e=rand(Layer3_hiddensize,1);

%target
y=X*A'*A';
tic
for t=1:iteration
    F1_hidden=X * A';
    for i=1:n
          Layer2_output(:,i) = Leaky_ReLU( Layer2_w * F1_hidden(:,i) - Layer2_b);
    end
    total_Layer2_output=sum(Layer2_output,2)/n;
    F2_hidden=Layer2_output * A';
    for i=1:n
          Layer3_output(:,i) = Leaky_ReLU( Layer3_w * F2_hidden(:,i) - Layer3_b);
    end
    total_Layer3_output=sum(Layer3_output,2)/n;

    for i=1:n
        if (Layer3_w * F2_hidden(:,i) - Layer3_b)<0
              temp_Layer3_e(:,i)=(y(:,i)-Layer3_output(:,i))*0.25;
        else
            temp_Layer3_e(:,i)=(y(:,i)-Layer3_output(:,i));
        end
    end
    Layer3_e=sum(temp_Layer3_e,2)/n;
    for i=1:n
        if Layer2_w * F1_hidden(:,i) - Layer2_b<0
              temp_Layer2_e(:,i)=Layer3_w' * Layer3_e * 0.25;
        else
            temp_Layer2_e(:,i)=Layer3_w' * Layer3_e;
        end
    end
    Layer2_e=sum(temp_Layer2_e,2)/n;
    
    Layer3_w = Layer3_w + LearningRate  * Layer3_e * total_Layer2_output';
    testX=sum(F1_hidden,2)/n;
    Layer2_w = Layer2_w + LearningRate * Layer2_e * testX';
    Layer3_b = Layer3_b - LearningRate * Layer3_e;
    Layer2_b = Layer2_b - LearningRate * Layer2_e;
    
    Loss=(y-Layer3_output).*(y-Layer3_output);
    EverySample_Loss=sum(Loss,1)/m;
    TotalLoss(t,:)=sum(EverySample_Loss)/n;
    
    %%%%adaptive learning rate%%%%%%%%%%%%%%%%%%%%
    Belta_R=0.98;
    Belta_E=1.005;
    alpha_max=0.02;%最大学习率
    if t>1
        if TotalLoss(t,:) >1.01*TotalLoss(t-1,:)
            LearningRate=Belta_R*LearningRate;
        elseif TotalLoss(t,:)<TotalLoss(t-1,:) & LearningRate<alpha_max
                LearningRate=Belta_E*LearningRate;
        else
                LearningRate=LearningRate;
        end
    end
end
toc
for i=1:n
    error=y(:,i)-Layer3_output(:,i);
    mse(i,:)=sum(error.*error);
end
[OF_value,index_number]=sort(mse);
auc = Measure_AUC(mse, ADLabels);
disp(auc)

ODA_AbnormalObject_Number=index_number(n-Abnormal_number+1:end,:);
ODA_NormalObject_Number=index_number(1:n-Abnormal_number,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Evalutaion techniques%%%%%%%%%%%%%%%%%%%%%%%%
[Real_NormalObject_Number,Real_Normal]=find(Label==0);
[Real_AbnormalObject_Number,Real_Abnormal]=find(Label==1);

TP=length(intersect(Real_AbnormalObject_Number,ODA_AbnormalObject_Number));
FP=length(Real_AbnormalObject_Number)-TP;
TN=length(intersect(Real_NormalObject_Number,ODA_NormalObject_Number));
FN=length(Real_NormalObject_Number)-TN;

%Acc
ACC=(TP+TN)/(TP+TN+FP+FN);
fprintf('准确率ACC= %8.5f\n',ACC*100)
%DR
DR=TP/(TP+FN);
fprintf('检测率DR= %8.5f\n',DR*100)
%P
P=TP/(TP+FP);
fprintf('查准率P= %8.5f\n',P*100)
%FAR
FAR=FP/(TN+FP);
fprintf('误报率FAR= %8.5f\n',FAR*100)

end

