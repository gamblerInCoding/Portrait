function dts(x)
a = x(:);
nans = isnan(a);
ind = find (nans); %nan是0/0.
a(ind)=[];
xbar= mean(a);
disp(['均值是：',num2str(xbar)]);
s2 = var(a);
disp(['方差是：',num2str(s2)]);
s = std(a);
disp(['标准差是：',num2str(s)]);%数据里必须是元素的类型一样，所以要有num2str()函数转一下。
R = range(a);
disp(['极差是：',num2str(R)]);
cv = 100*s./xbar;%它是一个相对的数且没有量纲，所以更具有说明性。
disp(['变异系数是：',num2str(cv)]);
g1 = skewness(a,0);
disp(['偏度：',num2str(g1)]);
g2=kurtosis(a,0);
disp(['峰度',num2str(g2)]);
end