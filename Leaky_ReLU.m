function y = Leaky_ReLU(x)
%LEAKY_RELU �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
a=2;
if x>=0
    y=x;
else
    y=x/a;
end
end

