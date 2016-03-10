function [ output_args ] = BanditDraw( input_args )
%BANDITDRAW Summary of this function goes here
%   Detailed explanation goes here

gamma = 0.8:0.02:1.0
regret = [26.861232487654906, 30.173629361521765, 28.06642065327448, 17.037036058789813, 14.225566468781789, 9.33868451228841, 7.840354008563545, 18.116092306737634, 41.63328616613417, 41.14669299784523, 75.40301350745894]

%X1 = load('new-regret1.txt');
%X2 = load('new-regret2.txt');
%X3 = load('new-exploit5.txt');

%x=1:size(X1,1);
%h=figure(1);
plot(gamma,regret,'r', 0.9, 9.33868451228841, 'k.','MarkerSize',20);
title('Network 6461');
xlabel('\gamma');
ylabel('Regret');
%legend('C^3-UCB', 'CombCascade');
%saveas(h,'f','eps');

end

