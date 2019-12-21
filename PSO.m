% Xiahua Liu
% PSO - Multi Gaussian Function

close all
clear
clc
disp('This program is written by Xiahua Liu')

%% Target function

% MATLAB would not pass function as an object
% So we wrap it up using lambda expression
target_function=@(x) multi_gauss_fn(x);

x_limit=[0,50];
y_limit=[0,50];

%% Parameters

XTICKS=50; % How many ticks on x direction in particle initialization process
YTICKS=50; % How many ticks on y direction in particle initialization process
PARTICLES=XTICKS*YTICKS; % How many particles in evolution
EPOCHS=500; % How many generations

INERTIA_MAX=0.9; % INERTIA decay overtime
INERTIA_MIN=0;

C1=0.1; % Local search weight
C2=0.2; % Global search weight


%% Initialize particles according to grids
x_grid=linspace(x_limit(1),x_limit(2),XTICKS);
y_grid=linspace(y_limit(1),y_limit(2),YTICKS);

% Particle information
particles=zeros(2,PARTICLES);
velocity=zeros(2,PARTICLES);

local_best_dot=zeros(2,PARTICLES);
local_best_value=inf(1,PARTICLES);

global_best_dot=[0;0];
global_best_value=inf;

for i=1:YTICKS
    for j=1:XTICKS
        particles(1,j+(i-1)*50)=x_grid(i);
        particles(2,j+(i-1)*50)=y_grid(j);
    end
end

count=0;

%% End evolving when the mean variance is very small
for i=1:200
    % Get decayed inertia every generation
    inertia=(INERTIA_MAX-INERTIA_MIN)*exp(-0.01*count)+INERTIA_MIN;
    count=count+1;
    
    % Get values and update local best and global best
    % This uses synchronous update, we have only one for loop
    % Increase evolving speed
    
    for i=1:length(particles)
        dot=particles(:,i);
        value=target_function(dot);
        
        % See if the new value is better
        if value<local_best_value(i)
            local_best_value(i)=value;
            local_best_dot(:,i)=dot;
        end
        
        if value<global_best_value
            global_best_value=value;
            global_best_dot=dot;
        end
        
        % Update velocity equation
        velocity(:,i)=inertia*velocity(:,i)+C1*rand()*(local_best_dot(:,i)-...
            dot)+C2*rand()*(global_best_dot-dot);
        
        % Step to next 
        particles(:,i)=dot+velocity(:,i);
    end
    disp(['Global minimal value is: ',num2str(global_best_value)])
    disp(['Global minimal point is: [', num2str(global_best_dot(1)),', '...
        ,num2str(global_best_dot(2)),']'])
end