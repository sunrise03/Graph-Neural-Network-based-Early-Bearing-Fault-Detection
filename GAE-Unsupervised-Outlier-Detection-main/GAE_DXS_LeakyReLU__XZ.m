function [outputArg1,outputArg2] = GAE_DXS_LeakyReLU__XZ(inputArg1,inputArg2)

%Setting
X=load('Normalization_wbc.txt')';% load dataset
A=load('wbc_A.txt'); %load graph
Label=load('Label_wbc.txt'); %load label
ADLabels=load('Label_wbc.txt');%load label for calculate AUC
iteration=100; % set the number of iteration 
LearningRate=0.00001; % set the learning rate
[m,n]=size(X); % get the  size of X, m represent the number of object in X,n represent the number of features
Abnormal_number=20;%the number of true outliers in X
Layer2_hiddensize=2; %  the number of neuron in hiddenLayer
Layer3_hiddensize=m;

% if user want to train again , please use those code
% Layer2_w=rand(Layer2_hiddensize,m);
% Layer3_w=rand(Layer3_hiddensize,Layer2_hiddensize);
% Layer2_b=rand(Layer2_hiddensize,1);
% Layer3_b=rand(Layer3_hiddensize,1);

% Load the parameters we have trained
Layer2_w=load('Layer2_w.txt');
Layer2_b=load('Layer2_b.txt');
Layer3_w=load('Layer3_w.txt');
Layer3_b=load('Layer3_b.txt');
Layer2_output=rand(Layer2_hiddensize,1);
Layer3_output=rand(Layer3_hiddensize,1);
Layer2_e=rand(Layer2_hiddensize,1);
Layer3_e=rand(Layer3_hiddensize,1);

% train goal
y=X*A'*A';

% start train
for t=1:iteration
   
    %Forward propagation
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
    
    %back propagation    
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
    
    Loss=(y-Layer3_output).*(y-Layer3_output);%Each row in loss represents the loss value of the sample of the corresponding row
    EverySample_Loss=sum(Loss,1)/m;%Average loss per sample
    TotalLoss(t,:)=sum(EverySample_Loss)/n;%Overall loss of all samples
    
    %%%%adaptive learning rate%%%%%%%%%%%%%%%%%%%%
    Belta_R=0.98;
    Belta_E=1.005;
    alpha_max=0.02;%max learning rate
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

for i=1:n
    error=y(:,i)-Layer3_output(:,i);
    mse(i,:)=sum(error.*error);%mse corresponds to the outlier factor of each object
end
[OF_value,index_number]=sort(mse);
testPlot=Layer2_output';%plot
auc = Measure_AUC(mse, ADLabels);
fprintf('AUC=',disp(auc));

ODA_AbnormalObject_Number=index_number(n-Abnormal_number+1:end,:);%The number of the outlier identified by the GAE
ODA_NormalObject_Number=index_number(1:n-Abnormal_number,:);%The number of the normal object identified by the GAE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Calculation of the actual detection rate/accuracy rate/false alarm rate and other evaluation indicators of the GAE%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Real_NormalObject_Number represents the number of the real normal object in the data set，Real_AbnormalObject_Number represents the number of the truly outliers in the data set
[Real_NormalObject_Number,Real_Normal]=find(Label==0);
[Real_AbnormalObject_Number,Real_Abnormal]=find(Label==1);

TP=length(intersect(Real_AbnormalObject_Number,ODA_AbnormalObject_Number));
FP=length(Real_AbnormalObject_Number)-TP;
TN=length(intersect(Real_NormalObject_Number,ODA_NormalObject_Number));
FN=length(Real_NormalObject_Number)-TN;

%ACC
ACC=(TP+TN)/(TP+TN+FP+FN);
fprintf('ACC= %8.5f\n',ACC*100)
%DR
DR=TP/(TP+FN);
fprintf('DR= %8.5f\n',DR*100)
%P
P=TP/(TP+FP);
fprintf('P= %8.5f\n',P*100)
%FAR
FAR=FP/(TN+FP);
fprintf('FAR= %8.5f\n',FAR*100)

%plot confusion matrix

% Confusion_matrix=[TP,FN;FP,TN];
% Figure_Confusion_matrix=heatmap(Confusion_matrix);
% figure(1)
% for j=0:iteration-1
%     j=j+1;
%     axis_x(j,:)=j;
% end
% plot(axis_x,TotalLoss,'LineWidth',2);
% hold on;



%scatter(testPlot(:,1),testPlot(:,2),25) ;
%hold on
%for i=1:Abnormal_number
%    scatter(testPlot(Real_AbnormalObject_Number(i,:),1),testPlot(Real_AbnormalObject_Number(i,:),2),50,'g','d','filled') ;
%   hold on
%end
end
