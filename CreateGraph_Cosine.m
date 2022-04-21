%load the dataset
%each row is an object.
X=load('group1.txt')';
k=20; %select the nearest neighbor number

Cos=pdist2(X,X,'cosine');
[m,n]=size(X);
[cos,index]=sort(Cos,2);
Similarity_k=index(:,2:k+1);
for i=1:m
    for j=1:k
        temp(i,j)=Cos(i,Similarity_k(i,j)); %The value of the degree of similarity between each object and its similarity
    end
end

for i=1:m
    for j=1:k
        A_value(i,j)=Cos(i,Similarity_k(i,j))/sum(temp(i,:),2);
    end
end

A=zeros(m,m);
for i=1:m
    for j=1:k
        A(i,Similarity_k(i,j))=A_value(i,j);
    end
end
temp_eye=eye(m,m);

A=A+temp_eye;

dlmwrite('D:\matlab2019a\matlab files\Machine_error\Model\GAE_COS\data_Adjacency_Matrix_cosine\A_20_group1.txt',A,' ');


