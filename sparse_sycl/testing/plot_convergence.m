function plot_convergence( write )

% This script visualizes the convergence for data stored in convergence.m
% For write == 1, the plot is written to disc as pdf.


% plot defaults
LW1 = 'linewidth'; lw1 = 1;
LW2 = 'linewidth'; lw2 = 2;
LW3 = 'linewidth'; lw3 = 2;
LW3 = 'linewidth'; lw4 = 3;
FS = 'fontsize'; fs = 16;
MS = 'markersize'; ms = 100;
MC = 'markerfacecolor'; mc = 'auto';

% set different color scheme
mycolors=[
         0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840
    0.2500    0.2500    0.2500
];

myblue    = mycolors(1,:);
myorange  = mycolors(2,:);
myyellow  = mycolors(3,:);
mymagenta = mycolors(4,:);
mygreen   = mycolors(5,:);
mycyan    = mycolors(6,:);
myred     = mycolors(7,:);
myblack   = mycolors(8,:);
% example: plot(x, y, '-o', 'color', myblue);



% load data
magma_output

h = figure(1);

semilogy(...
     data(:,1),data(:,2), 'x-', 'color', mygreen, ...
     LW2, lw2, MC, mc);
hold on;
ylabel('Residual norm',FS,fs);
xlabel('Number of iterations',FS,fs)
set(gca,FS,fs),grid on
%xlim([0 20])
%axis([0 10 exactiters-1 2000 ])
if( write == 1 )
    saveas(h, plotnameiters, 'pdf');
end
hold off;
