clear;
clc;
datas = importdata("train_dataset.csv");
data = getfield(datas, 'data');
name = getfield(datas, 'textdata');
disp("数据集形状")
size(data)
table = readtable("train_dataset.csv");
disp("数据集首5行数据")
head(table,5)
disp("数据集描述性统计")
str = input('请输入你要进行查看描述性统计信息的数据集列的名字：','s');
mark = 0;
for i = 1:30
    if strcmp(str,name(1,i))==1 
        mark = i;
    end
end
%调用函数对那列进行描述性统计   
dts(data(:,mark-1))
fws(data(:,mark-1))
%数据预处理
disp("数据预处理")
disp("去重")
size(data,1)
data=unique(data,'rows');
size(data,1)
disp("缺失值")
sum(isnan(data))
disp("异常值")
figure(1);boxplot(data(:,1:10))
title('变量1-10箱形图','fontsize',12)
figure(2);boxplot(data(:,11:20))
title('变量11-20箱形图','fontsize',12)
figure(3);boxplot(data(:,21:29))
title('变量21-29箱形图','fontsize',12)
% 使用肖维勒方法（等置信概率）剔除异常值
[m n] = size(data);
Y = [];            
w = 1 + 0.4*log(m);    % 肖维勒系数（近似计算公式）
for i = 1:n
   x = data(:,i);    
   YiChang = abs(x-mean(x)) > w*std(x);
   Y(:,i) = YiChang;
end
[u v] = find(Y() == 1);   % 找出异常值所在的行与列
ls = size(u,1);
uu = unique(u);    % 剔除重复的行数
now = size(uu,1);
disp("剔除异常值的数量:");
size(uu,1)
data(uu,:) = [ ];   %令异常值所在行为空,即剔除异常值
%数据可视化
disp("数据可视化")
disp("用户话费敏感度")
target=data(:,29);
phoneprice=data(:,13);
result = tabulate(phoneprice(:));
lab = num2str(result(:,1));
figure(4)
subplot(1,2,1)
bar(result(:,1),result(:,2))
title('用户话费敏感度分布条形图','fontsize',12)
subplot(1,2,2)
pie(result(:,2))
title('用户话费敏感度分布饼状图','fontsize',12)
legend(lab);
disp("用户最近一次缴费距今时长（月）")
ms = [0,0];
timetonow=data(:,7);
for i=1:46645
    if timetonow(i,1)==0
        ms(1,1)=ms(1,1)+target(i,1);
    end
    if timetonow(i,1)==1
        ms(1,2)=ms(1,2)+target(i,1);
    end
end
result = tabulate(timetonow(:));
for i=1:2
    ms(1,i)=ms(1,i)/result(i,2);
end
lab = num2str(result(:,1));
figure(5)
subplot(1,3,1)
bar(result(:,1),result(:,2))
title('用户最近一次缴费距今时长分布条形图','fontsize',12)
subplot(1,3,2)
pie(result(:,2))
title('用户最近一次缴费距今时长分布饼状图','fontsize',12)
legend(lab);
subplot(1,3,3)
plot(result(:,1),ms,'-*r');
title('用户最近一次缴费距今时长分布与信用分关系折线图','fontsize',12)
disp("用户实名制是否通过核实")
ident=data(:,1);
result = tabulate(ident(:));
lab = num2str(result(:,1));
figure(6)
subplot(1,2,1)
bar(result(:,1),result(:,2))
title('用户实名制是否通过核实分布条形图','fontsize',12)
subplot(1,2,2)
pie(result(:,2))
title('用户实名制是否通过核实分布饼状图','fontsize',12)
legend(lab);
%用户是否经常逛商场
figure(7)
shop=data(:,15);
disp("用户是否经常逛商场")
ms = [0,0];
timetonow=shop
for i=1:46645
    if timetonow(i,1)==0
        ms(1,1)=ms(1,1)+target(i,1);
    end
    if timetonow(i,1)==1
        ms(1,2)=ms(1,2)+target(i,1);
    end
end
result = tabulate(timetonow(:));
for i=1:2
    ms(1,i)=ms(1,i)/result(i,2);
end
lab = num2str(result(:,1));

subplot(1,3,1)
bar(result(:,1),result(:,2))
title('用户是否经常逛商场分布条形图','fontsize',12)
subplot(1,3,2)
pie(result(:,2))
title('用户是否经常逛商场分布饼状图','fontsize',12)
legend(lab);
subplot(1,3,3)
plot(result(:,1),ms,'-*r');
title('用户是否经常逛商场关系折线图','fontsize',12)
disp("用户实名制是否通过核实")
figure(8)
shop=data(:,10);
scatter(shop,target)
xlabel('用户账单当月总费用')
ylabel('信用分')
title('用户账单当月总费用与信用分之间的散点图','fontsize',12)
figure(9)
shop6=data(:,9);
plot(shop6,target,'-og')
xlabel('用户近6个月平均消费值')
ylabel('信用分')
title('用户近6个月平均消费值与信用分之间关系的折线图','fontsize',12)
figure(10)
year=data(:,2);
result = zeros(1,10);
for i=1:46645
    result(1,floor(year(i,1)/10)+1)=result(1,floor(year(i,1)/10)+1)+1;
end
pareto(result);
xlabel('用户年龄')
ylabel('人数')
figure(11)
[f1, x1] = ksdensity(year);
t1 = area(x1, f1);t1.FaceColor='b'; t1.FaceAlpha=0.5;
title('用户年龄核密度分布图');
xlabel("年龄");
figure(12)
rho = corr(data);
string_name={'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29'};
xvalues = string_name;
yvalues = string_name;
h = heatmap(xvalues,yvalues,rho, 'FontSize',10, 'FontName','Times New Roman');
h.Title = 'Correlation Coefficient';
colormap(jet)
%特征工程
disp('特征创造')
newFea = zeros(46645,1);
newFea(:,1)=data(:,8)-data(:,9);
data = [data newFea];
newFea(:,1)=data(:,10)-data(:,9);
data = [data newFea];
disp('特征筛选')
data(:,1) = [];
disp('特征标准化')
traindata = data;
traindata(:,28) = [];
newData = zscore(traindata);
thetarget = zscore(target);
figure(13)
disp('特征降维')
[COEFF,SCORE,latent,tsquared,explained,mu]=pca(newData);
data_PCA=newData*COEFF(:,1:10);
latent1=100*latent/sum(latent);%将latent总和统一为100，便于观察贡献率
pareto(latent1);%调用matla画图 pareto仅绘制累积分布的前95%，因此y中的部分元素并未显示
xlabel('Principal Component');
ylabel('Variance Explained (%)');
iris_pac=data_PCA(:,1:5) ;
figure(14)
[COEFF1,SCORE1,latent1,tsquared1,explained1,mu1]=pca(traindata);
data_PCA1=traindata*COEFF1(:,1:11);
latent11=100*latent1/sum(latent1);%将latent总和统一为100，便于观察贡献率
pareto(latent11);%调用matla画图 pareto仅绘制累积分布的前95%，因此y中的部分元素并未显示
xlabel('Principal Component');
ylabel('Variance Explained (%)');
iris_pac1=data_PCA1(:,1:11) ;
% %建立模型
% train_x = iris_pac(1:size(iris_pac,1)*0.75,:);
% train_y = thetarget(1:size(thetarget,1)*0.75,:);
% test_x = iris_pac(size(iris_pac,1)*0.75:end,:);
% test_y = thetarget(size(thetarget,1)*0.75:end,:);
% [b,bint,r,rint,stats] = regress(train_y,train_x);
% disp(['y =', num2str(b(1)),'+',num2str(b(2)),'*x1','+',num2str(b(3)),'*x2+',...
%     num2str(b(4)),'*x3+', num2str(b(5)),'*x4'])
% %模型评价-rmse
% predict = [];
% rmse = 0;
% for i=1:11662
%     predict(i,1)=b(1)+b(2)*test_x(i,1)+b(3)*test_x(i,2)^2+b(4)*test_x(i,3)^3+b(5)*test_x(i,4)^4;
%     rmse = rmse + (predict(i,1)-test_y(i,1))^2;
% end
% disp("rmse值:")
% disp(rmse)
%决策树回归模型
figure(15)
iris_pac0 = iris_pac1
target0 = target
assess_x = iris_pac1(size(iris_pac1,1)*0.75:end,:);
assess_y = target(size(target,1)*0.75:end,:);
iris_pac1 = iris_pac1(1:size(iris_pac1,1)*0.75,:);
target = target(1:size(target,1)*0.75,:);
leafs = logspace(1,2,10);
rng('default')
N = numel(leafs);
err = zeros(N,1);
for n=1:N
        t =  fitrtree(iris_pac1,target,'CrossVal','On', 'MinLeaf',leafs(n));%交叉验证法估计算法精度
        err(n) = kfoldLoss(t);%计算误差
end
plot(leafs,err);
grid on
xlabel('Min Leaf Size');
ylabel('cross-validated error');
title('决策树minleaf调参')
OptimalTree = fitrtree(iris_pac1,target,'minleaf',46);
resuberror = resubLoss(OptimalTree) %衡量误差，默认均方差算法，此处可以设置损失函数
lossOpt = kfoldLoss(crossval(OptimalTree))
Ynew = predict(OptimalTree,iris_pac1);
trainResult = fix(Ynew);
newYnew = predict(OptimalTree,assess_x);
testResult = fix(newYnew);
rmse=0;
trainResult1 = trainResult;
testResult1 = testResult;
for i=1:34983
    rmse = rmse + (trainResult(i,1)-target(i,1))^2;
end
rmse=rmse/34983;
rmse1= 0;
for i=1:11662
    rmse1 = rmse1 + (testResult(i,1)-assess_y(i,1))^2;
end
rmse1=rmse1/11662;
disp("决策树回归训练集上的rmse:")
disp(rmse)
disp("决策树回归测试集上的rmse:")
disp(rmse1)






%高斯回归模型

t =  fitrgp(iris_pac1,target);
Ynew = predict(t,iris_pac1);
trainResult = fix(Ynew);
newYnew = predict(t,assess_x);
testResult = fix(newYnew);
rmse=0;
trainResult2 = trainResult;
testResult2 = testResult;
for i=1:34983
    rmse = rmse + (trainResult(i,1)-target(i,1))^2;
end
rmse=rmse/34983;
rmse1= 0;
for i=1:11662
    rmse1 = rmse1 + (testResult(i,1)-assess_y(i,1))^2;
end
rmse1=rmse1/11662;
disp("高斯回归训练集上的rmse:")
disp(rmse)
disp("高斯回归测试集上的rmse:")
disp(rmse1)






%线性回归模型

t =  fitrlinear(iris_pac1,target);
Ynew = predict(t,iris_pac1);
trainResult = fix(Ynew);
newYnew = predict(t,assess_x);
testResult = fix(newYnew);
rmse=0;
trainResult3 = trainResult;
testResult3 = testResult;
for i=1:34983
    rmse = rmse + (trainResult(i,1)-target(i,1))^2;
end
rmse=rmse/34983;
rmse1= 0;
for i=1:11662
    rmse1 = rmse1 + (testResult(i,1)-assess_y(i,1))^2;
end
rmse1=rmse1/11662;
disp("线性回归训练集上的rmse:")
disp(rmse)
disp("线性回归测试集上的rmse:")
disp(rmse1)



%模型融合-stacking/回归树集成
trainDatas = [trainResult1 trainResult2 trainResult3];
testDatas = [testResult1 testResult2 testResult3];

t =  fitrensemble(trainDatas,target);
Ynew = predict(t,trainDatas);
trainResult = fix(Ynew);
newYnew = predict(t,testDatas);
testResult = fix(newYnew);
rmse=0;
for i=1:34983
    rmse = rmse + (trainResult(i,1)-target(i,1))^2;
end
rmse=rmse/34983;
rmse1= 0;
for i=1:11662
    rmse1 = rmse1 + (testResult(i,1)-assess_y(i,1))^2;
end
rmse1=rmse1/11662;
disp("模型融合训练集上的rmse:")
disp(rmse)
disp("模型融合测试集上的rmse:")
disp(rmse1)