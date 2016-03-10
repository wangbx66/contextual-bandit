function [ output_args ] = BanditDraw( input_args )
%BANDITDRAW Summary of this function goes here
%   Detailed explanation goes here

X1 = load('new-regret1.txt');
X2 = load('new-regret2.txt');
%X3 = load('new-exploit5.txt');

x=1:size(X1,1);
%h=figure(1);
plot(x,X1,'r',x,X2,'k');
title('Synthetic Data');
xlabel('Time t');
ylabel('Regret');
legend('C^3-UCB', 'CombCascade');
%saveas(h,'f','eps');

end

