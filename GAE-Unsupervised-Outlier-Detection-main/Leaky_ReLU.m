function y = Leaky_ReLU(x)
a=2;
if x>=0
    y=x;
else
    y=x/a;
end
end

