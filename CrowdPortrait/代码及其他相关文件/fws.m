function fws(x)
a = x(:);
a(isnan(a))=[];
ss5 = prctile(a,50);
disp(['中位数是：',num2str(ss5)]);
ss25 = prctile(a,25);
disp(['下四分位数是：',num2str(ss25)]);
ss75 = prctile(a,75);
disp(['上四分位数是：',num2str(ss75)]);
RS = ss75-ss25;
disp(['四分位极差：',num2str(RS)]);
end