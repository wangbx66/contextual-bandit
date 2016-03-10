function [ output_args ] = BanditDraw( input_args )
%BANDITDRAW Summary of this function goes here
%   Detailed explanation goes here

X1 = load('new-exploit1.txt');
%X2 = load('new-exploit3.txt');
X3 = load('new-exploit4.txt');

x=1:size(X1,1);
%h=figure(1);
plot(x,X1,'r',x,X3,'k--');
title('Network 1755');
xlabel('Time t');
ylabel('Reward');
legend('C^3-UCB', 'CombCascade');
%saveas(h,'f','eps');

end

