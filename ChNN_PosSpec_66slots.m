% back propagation general

feature('DefaultCharacterSet','UTF-8'); % For MAC usage
tic
clc


%% Load data (includes the actual words (not binray!), their frequency, and their semantic labels in binary vector form
%load PosSpec_materials.csv  % This file includes all the necessary information, including input and output codings. Specifically, it should include:
% 1. The input characters
% 2. The frequency of the characters
% 3. The desired output label vector-representation
% 4. The input representation
% Columns are: 22[material information] + 66(components) + 4(Structure) + 23(initial) + 33(final) + 5(tone)

newRun = 1; % 0: continue run that was previously ended and saved; 1: Start training from scratch;

d = readtable('./1 PosSpec_materials.csv');
d = d(d.CHR_million>0, :);    % size(d) [5209, 153]

characters = d(:, 1:22);
freq = table2array(d(:, 4));
x = transpose(table2array(d(:, 23:92)));
y0 = transpose(table2array(d(:, 93:end)));

fff = length(freq)*log10(freq+1)/sum(log10(freq+1));
fff = repmat(fff',size(y0,1),1);


%% control parameters:
if newRun
    doubleRoute = 0; % decide whether we use separate identity and position pathways, or not
    reached = 0; t = 2;
    beta=1; % slope of exponential function. Always 1 in all simulations I used.
    method = 7; % method of coding. always 7 in all current simulations
    L = 4; % number of layers
    N = zeros(L,1);
    momGo = 0; % initialize Go/noGo parameter determining when momentum and db are applied. % Should always be 0 for a new run.
    Start_mom = 1200; % determine at what epoch momentum and dbd will start. Use 0 for never.
    runs =300000; % Number of runs
    alpha = 0.9; % Momentum parameter
    dec = 0.00001; % Decay parameter
    etta0 = 0.00001; %Global learning rate
    theta = 0; % delta-bar-delta exponential averaging factor
    kappa = 0.1; % delta-bar-delta additivie factor
    phi = 0.1; % delta-bar-delta multiplication factor
    
    switch L
        case 4, N(1) = size(x,1);  N(2) = 2000; N(3) = 200; N(4)=size(y0,1); % Number of units in each layer
        case 3, N(1) = size(x,1);  N(2) = 3000; N(3) = 200;  % Number of units in each layer
    end
    
    P = size(x,2); % Number of samples
    
    %% initial weights (including bias-neurons), deltaBar (used by dbd) and etta (local learning late, used by dbd):
    clear W s
    
    W = cell(1,L-1);
    dW = cell(1,L-1);
    deltaBar = cell(1,L-1);
    etta = cell(1,L-1);
        
   
    for i=1:L-1
        W{i} = 0.2*rand(N(i+1),N(i)+1) - 0.1;
        dW{i} = zeros(N(i+1),N(i)+1);
        deltaBar{i} = dW{i};
        etta{i} = dW{i} +1;
        if i==1 && doubleRoute==1 % if separate identity and position pathways are used, W needs to be adjusted
            W{i}(1:N(2)/2,totalL+1:N(1)) = 0; %% Zian: set ID slots to 201-400 L1 nodes connections as 0
            W{i}(N(2)/2+1:N(2),1:totalL) = 0; %% Zian: set POS slots to 1-200 L1 nodes connections as 0
        end
        W{i} = W{i}*0.1; % Decrease initial weights further in order to prevent initial pinning of units        
    end
    W{1}(:,end) = -1.73; % specific intial negative values for biases
    W{2}(:,end) = -1.73; % specific intial negative values for biases
    if L==4, W{3}(:,end) = -1.73; end % specific intial negative values for biases [0.04886 for ChNN]
    
    % initial output of all examples and new examples:
    s = cell(1,L);
    s{1} = x;
    for i=2:L
        s{i} = 1./(1+exp(-(beta*W{i-1}*[s{i-1}; ones(1,size(s{i-1},2))])));
    end
    %sKeep = cell(10,1); sKeep{1} = s;
        
    % Zian: initial momTerm, delta
    momTerm = cell(1,L-1);
    delta = cell(1,L);
    delta{L} = (s{L}-y0).*fff*beta;
    delta_tmp = cell(1,L-1);
    for i=1:L-1
        momTerm{i} = dW{i};
        delta_tmp{i} = zeros((N(i)+1), P);
        delta{i} = zeros(N(i), P);
    end    
end

%% initial training error and other paramaters:
global Etrain;
global Etrain2;
if newRun
    Etrain = zeros(1,runs); % Etrain is mean of MSE (mean squared error) across all output nodes
    Etrain(1) = mean(0.5*sum(((y0-s{end}).^2),2)/P);
    Etrain2 = zeros(1,runs); % Etrain2 is Cross Entropy error
    Etrain2(1) = mean(-mean((y0.*log(s{end})+(1-y0).*log(1-s{end})),1)); tic
    % Initialize Blair's parameters:
    %Wcos = zeros(1,runs); Wcos_tmp = [W{1}(:);W{2}(:); W{3}(:)]; % Wcos is the cos measure; Wcos_tmp will be used during its calculation
    %Wabs = zeros(1,runs); Wabs(1) = mean(abs([W{1}(:);W{2}(:); W{3}(:)])); % absolute value of concatenated weights
    
    %Sdiv1 = zeros(1,runs); Sdiv1(1) = mean(mean(abs(s{end-2}(1:N(2),:)-0.5))); % 1st hidden layer's unit deviation from 0.5
    %Sdiv2 = zeros(1,runs); Sdiv2(1) = mean(mean(abs(s{end-1}-0.5)));   % 2nd hidden layer's unit deviation from 0.5
    
    %eg_mean = zeros(1,runs); eg_mean(1) = 1; % mean local error-rate (for dbd)
    stopC = zeros(1,runs);
    stopC_eg = zeros(1,runs);
end

%plot(Etrain); drawnow; set(gcf, 'KeyPressFcn', @myKeyPressFcn) % used for allowing the user to pause the simulation and look at current Error



%% Learning loop:
while t<=runs && reached==0
    if t/100==floor(t/100), toc; disp(t); end % display progress every 10 iterations
%     if t/25000==floor(t/25000)
%         disp('Saving temporal data'); save tempDATA;
%     end
    % initialize dWs
    
    for i=1:L-1
        momTerm{i} = dW{i}; % this parameter keeps the last dW values
    end
    
    % Compute delta and dW of last layer
    delta{L} = (s{end}-y0).*fff*beta; %% Zian: error of each word scaled to fff
    dW{L-1} = -etta0*delta{L}*[s{L-1}; ones(1,P)]';
    
    % Compute rest of deltas and dWs, using backprop:
    for i=L-1:-1:2
        delta_tmp{i} = (W{i}'*delta{i+1}).*[s{i}.*(1-s{i})*beta; ones(1,P)];
        delta{i} = delta_tmp{i}(1:end-1,:);
        dW{i-1} = -etta0*delta{i}*[s{i-1}; ones(1,P)]';
    end
    
    if t==Start_mom, momGo = 1; end % allow momentum and dbd to begin
    
    % Update weights and local error-rates of the dbd algorithm:
    for i=1:L-1
        etta{i} = etta{i}+momGo*(kappa*((deltaBar{i}.*dW{i})>0)-phi*etta{i}.*((deltaBar{i}.*dW{i})<0)); % dbd algorithm
        deltaBar{i} = (1-theta)*dW{i}+theta*deltaBar{i}; % used by dbd algorithm
        W{i} = (1-dec)*W{i}+etta{i}.*dW{i}+momGo*alpha*momTerm{i};
        if i==1 && doubleRoute==1
            W{i}(1:N(2)/2,totalL+1:N(1)) = 0;
            W{i}(N(2)/2+1:N(2),1:totalL) = 0;
        end
        %W{i} = (1-dec)*W{i}+dW{i}+momGo*alpha*momTerm{i};
    end
    
    % neuron values for all examples and new examples:
    for i=2:L
        s{i} = 1./(1+exp(-(beta*W{i-1}*[s{i-1}; ones(1,size(s{i-1},2))])));
    end
    
    %if t<=10, sKeep{t} = s; end
    
    % Update error:
    Etrain(t) = mean(0.5*sum(((y0-s{end}).^2),2)/P);
    Etrain2(t) = mean(-nanmean((y0.*log(s{end})+(1-y0).*log(1-s{end}))));
    
    % Update Blair's parameters:
    %tmp = [dW{1}(:); dW{2}(:); dW{3}(:)]; Wcos(t-1) = tmp'*Wcos_tmp/(norm(tmp)*norm(Wcos_tmp)); Wcos_tmp = tmp;
    %Wabs(t) = mean(abs([W{1}(:);W{2}(:); W{3}(:)]));
    %Sdiv1(t) = mean(mean(abs(s{end-2}(1:N(2),:)-0.5)));
    %Sdiv2(t) = mean(mean(abs(s{end-1}-0.5)));
    %eg_mean(t) = mean([etta{1}(:); etta{2}(:); etta{3}(:)]);
    
    % stopping criterion:
    if t/200==floor(t/200)
        %%Zian: if any value in a column of s{end} is larger or smaller than
        %%the corresponding y0 by 0.5, (sign(abs(y0-s{end})-0.5)+1)/2 would
        %%be 1, then that whole column would be 1 in the calculation of
        %%sum(sign(sum((sign(abs(y0-s{end})-0.5)+1)/2))). So only if all
        %%values are within +/-0.5 of their target values, the stopC(t) can
        %%be 1.
        stopC(t) = (P-sum(sign(sum((sign(abs(y0-s{end})-0.5)+1)/2))))/P;
        eg_old = mean(Etrain2(t-199:t-100)); eg_new = mean(Etrain2(t-99:t));
        stopC_eg(t) = (eg_new-eg_old)/eg_new;
        disp([stopC(t) abs(stopC_eg(t))]);
        if stopC(t)>0.9995 && abs(stopC_eg(t))<0.05
            disp('Stopped'); reached = 1;
        end
    end
    drawnow
    t = t+1;
end
%% Figures:
% maxStep = 45000;
% plot(Etrain2(1:maxStep)); hold on; plot(Wcos(1:maxStep),'r'); plot(Etrain2(1:maxStep)); legend('CEE','cos');
% figure(2); plot(Sdiv1(1:maxStep)); hold on; plot(Sdiv2(1:maxStep),'r'); legend('1st hid. deviation','2nd hid. deviation')
% figure(3); plot(eg_mean(1:maxStep)); legend('local error-rate mean');
% figure(4); hist(s{2}(:),100); legend('1st hidden layer activation histogram (final)')
% figure(5); hist(s{3}(:),100); legend('2nd hidden layer activation histogram (final)')

save(['ChNN_PosSpec_66slots_' date], 'characters', 'y0', 'W', 'L', 't', 'Etrain', 'Etrain2', 'etta0', 'etta', 'stopC', 'stopC_eg');