function y = Leaky_ReLU(x)
%LEAKY_RELU 此处显示有关此函数的摘要
%   此处显示详细说明
a=2;
if x>=0
    y=x;
else
    y=x/a;
end
end

