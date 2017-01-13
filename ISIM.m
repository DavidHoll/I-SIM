function [data, Score, inHeader, input, LagTime, Lim] =...
    ISIM(data, timestamp, Biomet, Biomet_header,...
    FuzzyFP, FuzzyFP_header, datasetname)
%% initialize Progress Window

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'initializing...';
dim = [300 200];
iconsClassName = 'com.mathworks.widgets.BusyAffordance$AffordanceSize';
iconsSizeEnums = javaMethod('values',iconsClassName);
SIZE_32x32 = iconsSizeEnums(2);  % (1) = 16x16,  (2) = 32x32
jObj = com.mathworks.widgets.BusyAffordance(SIZE_32x32, BusyText);  % icon, label
jObj.setPaintsWhenStopped(true);  % default = false
jObj.useWhiteDots(false);
Prog_wndw = figure('pos', [500 500 dim(1) dim(2)],...
    'menubar', 'none',...
    'name', ['Progress, ' datasetname],...
    'NumberTitle', 'off');
javacomponent(jObj.getComponent, [0,0,dim(1),100], gcf);
jObj.start;

List = ['<b>Create input matrices (' num2str(0) '%)</b><br\>'...
    'Create scoring table<br\>'...
    'Parameterize MLP<br\>'...
    'Parameterize RBN<br\>'...
    'Run models'];
updateProgessWindow(List)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

%% create model input matrices I
% unlagged
input.unl = [Biomet.unl;FuzzyFP];
inHeader.unl = [Biomet_header.unl;FuzzyFP_header];
%% find lagtime with cross correlation (Kettunen 1996) and create lagged dataset

BusyText = 'Finding timelags...';
jObj.setBusyText(BusyText)

Biomet.lag = Biomet.unl;
LagTime = double(size(Biomet.unl));

for k = 1 : size(Biomet.unl,1)
    
    [~,LagTime(k),~,~] = lagtime_30min_v3(data.ModelIn, Biomet.unl(k,:)', .5, [.7 .7 .7]); %hold on
    
    if LagTime(k) > 0
        Biomet.lag(k,:) = [Biomet.unl(k,LagTime(k)*2+1:end) nan(1,LagTime(k)*2)];
    elseif LagTime(k) < 0
        Biomet.lag(k,:) = [nan(1,abs(LagTime(k))*2) Biomet.unl(k,1:end-abs(LagTime(k))*2)];
    end
end
Biomet.lag = Biomet.lag(LagTime~=0,:);
Biomet_header.lag = Biomet_header.unl(LagTime~=0);
Biomet_header.lag = cellfun(@(x) [x ' #lag#'], Biomet_header.lag, 'uniformoutput',0);

%% create model input matrices II
% lagged

input.lag = [Biomet.lag;FuzzyFP];

inHeader.lag = [Biomet_header.lag;FuzzyFP_header];

% lagged & unlagged
Biomet.both = [Biomet.unl; Biomet.lag];
Biomet_header.both = [Biomet_header.unl; Biomet_header.lag];
input.both = [Biomet.unl; Biomet.lag; FuzzyFP];
inHeader.both = [Biomet_header.unl; Biomet_header.lag; FuzzyFP_header];

%% run MLRs create scoring table

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Stepwise multilinear regression';
jObj.setBusyText(BusyText)
List = ['Create input matrices (' num2str(100) '%)<br\>'...
    '<b>Create scoring table(' num2str(0) '%)</b><br\>'...
    'Parameterize MLP<br\>'...
    'Parameterize RBN<br\>'...
    'Run models'];
updateProgessWindow(List)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

out = plotStepReg_v4(data.ModelIn, input.unl, inHeader.unl, timestamp,'all unlagged inputs');
data.MLR.unl = out; clear out
Score.unl.MLR = double(data.MLR.unl.inmodel);

out = plotStepReg_v4(data.ModelIn, input.lag, inHeader.lag, timestamp,'all lagged inputs');
% out = rmfield(out, 'unl');
data.MLR.lag = out; clear out
Score.lag.MLR = double(data.MLR.lag.inmodel);

out = plotStepReg_v4(data.ModelIn, input.both, inHeader.both, timestamp,'all lagged and unlagged inputs');
% out = rmfield(out, {'unl','lag'});
data.MLR.both = out; clear out
Score.both.MLR = double(data.MLR.both.inmodel);


%% AIC | UNL

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
substeps = 12;
BusyText = 'Finding #HLN for MLP with input matrix 1/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

% number of free parameters k (= no of weights):
% (no of inputs * no of hidden layer neurons) + hidden layer neurons
% AIC = n * ln(sse/n) + 2k
% AIC = n * ln(mse) + 2k
MaxNodes = 20;
iterations = 100;
valchecks = 6;
trainPerc = 70;
valPerc = 30;
testPerc = 0;

for m = 1 : MaxNodes
    
    nodes = m;
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((m/MaxNodes)*100) '%, 1/' num2str(substeps) ')</b><br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    freeParams = (m * size(input.unl,1)) + m;
    for k = 1:iterations
        [data.MLP.unl.output(k,:), data.MLP.unl.tr(k,m),data.MLP.unl.net{k,m}, ~, ~] =...
            genMLP_v2(input.unl,data.ModelIn,nodes,valchecks,trainPerc,valPerc,testPerc);
    end
    [data.MLP.unl.EnsembleOut(:,m),~,~,data.MLP.unl.EnsembleStats.r(m),~,data.MLP.unl.EnsembleStats.rmse(m),~] =...
        nNetIterationsMean(data.MLP.unl.output', data.ModelIn);
    
    data.MLP.unl.NodeTest.AIC(m) =...
        sum(~isnan(data.ModelIn)) * log(data.MLP.unl.EnsembleStats.rmse(m)^2) +...
        2*freeParams;
    
    % use bias adjustment term (AICc) if n/k < 40
    
    if (sum(~isnan(data.ModelIn)))/freeParams < 40
        data.MLP.unl.NodeTest.AIC(m) = data.MLP.unl.NodeTest.AIC(m) +...
            ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
        data.MLP.unl.NodeTest.AICc(m) = true;
    else
        data.MLP.unl.NodeTest.AICc(m) = false;
    end
end

% find minimum AIC
[data.MLP.unl.NodeTest.AIC_min, data.MLP.unl.NodeTest.minNodes]= min(data.MLP.unl.NodeTest.AIC);

% alternative way to find AIC optimum by fitting a 1/x
% function. #HLN is where 1st derivative goes towards zero.

[data.MLP.unl.NodeTest.AsympFit,...
    data.MLP.unl.NodeTest.AsympFit.AIC_min, data.MLP.unl.NodeTest.AsympFit.minNodes] = ...
    AsymptoteFit10(data.MLP.unl, MaxNodes);

% also fit parabola to recognize if #HLN follows u-shaped function.
% avoid selection of minNodes smaller than minimum of parabola by this

[data.MLP.unl.NodeTest.ParabFit.minimum,...
    data.MLP.unl.NodeTest.ParabFit.gof,...
    data.MLP.unl.NodeTest.ParabFit.fitresult] = ...
    fitParabola(data.MLP.unl.NodeTest.AIC, 1:MaxNodes);

if data.MLP.unl.NodeTest.minNodes > data.MLP.unl.NodeTest.AsympFit.minNodes
    if ~isempty(data.MLP.unl.NodeTest.ParabFit.minimum) &&...
            ismember(data.MLP.unl.NodeTest.minNodes,...
            round(data.MLP.unl.NodeTest.ParabFit.minimum) - 1:...
            round(data.MLP.unl.NodeTest.ParabFit.minimum) + 1) &&...
            data.MLP.unl.NodeTest.minNodes < MaxNodes
        data.MLP.unl.NodeTest.AsympFit.flag = false;
        data.MLP.unl.NodeTest.ParabFit.flag = true;
    else
        data.MLP.unl.NodeTest.minNodes = data.MLP.unl.NodeTest.AsympFit.minNodes;
        data.MLP.unl.NodeTest.AIC_min = data.MLP.unl.NodeTest.AIC(data.MLP.unl.NodeTest.minNodes);
        data.MLP.unl.NodeTest.AsympFit.flag = true;
        data.MLP.unl.NodeTest.ParabFit.flag = false;
    end
else
    data.MLP.unl.NodeTest.AsympFit.flag = false;
    data.MLP.unl.NodeTest.ParabFit.flag = false;
end


%% Akaike weights and evidence ratio
data.MLP.unl.NodeTest.deltaI = data.MLP.unl.NodeTest.AIC - data.MLP.unl.NodeTest.AIC_min;
data.MLP.unl.NodeTest.L_est = exp(-0.5.*data.MLP.unl.NodeTest.deltaI);
data.MLP.unl.NodeTest.w_i = data.MLP.unl.NodeTest.L_est ./ sum(data.MLP.unl.NodeTest.L_est);
data.MLP.unl.NodeTest.evidence_ratio = data.MLP.unl.NodeTest.w_i(data.MLP.unl.NodeTest.minNodes)./data.MLP.unl.NodeTest.w_i;
% semilogy(evidence_ratio), grid on, grid minor

%% sensitivity analysis schmidt 2008| UNL
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Sensitivity test (Schmidt 2008) 1/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

replace_perc = 100;
iterations = 100;
range = .8:.1:1.3;

for k = 1:iterations
    [data.sens.unl.RE100.out(:,k),~] = getRE_v4(input.unl, data,replace_perc,'MLP', data.MLP.unl.NodeTest.minNodes); %v4 -> replace with median
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/(iterations*2))*100) '%, 2/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end
Score.unl.RE100 = zeros(1,length(inHeader.unl));
for k = 1:length(inHeader.unl)
    [counts, ~] = hist(data.sens.unl.RE100.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.unl.RE100(k) = 1;
    end
end
replace_perc = 50;
for k = 1:iterations
    [data.sens.unl.RE50.out(:,k),~] = getRE_v4(input.unl, data,replace_perc,'MLP', data.MLP.unl.NodeTest.minNodes); %v4 -> replace with median
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str(((iterations+k)/(iterations*2))*100) '%, 3/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
end
Score.unl.RE50 = zeros(1,length(inHeader.unl));
for k = 1:length(inHeader.unl)
    [counts, ~] = hist(data.sens.unl.RE50.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.unl.RE50(k) = 1;
    end
end

%% AIC | LAG
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding #HLN for MLP with input matrix 2/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||


MaxNodes = 20;
iterations = 100;
valchecks = 6;
trainPerc = 70;
valPerc = 30;
testPerc = 0;

for m = 1 : MaxNodes
    
    nodes = m;
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((m/MaxNodes)*100) '%, 4/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    freeParams = (m * size(input.lag,1)) + m;
    for k = 1:iterations
        
        [data.MLP.lag.output(k,:), data.MLP.lag.tr(k,m),data.MLP.lag.net{k,m}, ~, ~] =...
            genMLP_v2(input.lag,data.ModelIn,nodes,valchecks,trainPerc,valPerc,testPerc);
    end
    
    [data.MLP.lag.EnsembleOut(:,m),~,~,data.MLP.lag.EnsembleStats.r(m),~,data.MLP.lag.EnsembleStats.rmse(m),~] =...
        nNetIterationsMean(data.MLP.lag.output', data.ModelIn);
    
    data.MLP.lag.NodeTest.AIC(m) =...
        sum(~isnan(data.ModelIn)) * log(data.MLP.lag.EnsembleStats.rmse(m)^2) +...
        2*freeParams;
    
    % use bias adjustment term (AICc) if n/k < 40
    
    if (sum(~isnan(data.ModelIn)))/freeParams < 40
        data.MLP.lag.NodeTest.AIC(m) = data.MLP.lag.NodeTest.AIC(m) +...
            ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
        data.MLP.lag.NodeTest.AICc(m) = true;
        
    else
        data.MLP.lag.NodeTest.AICc(m) = false;
    end
end

% find minimum AIC
[data.MLP.lag.NodeTest.AIC_min, data.MLP.lag.NodeTest.minNodes]= min(data.MLP.lag.NodeTest.AIC);

% alternative way to find AIC optimum by fitting a 1/x
% function. #HLN is where 1st derivative goes towards zero.

[data.MLP.lag.NodeTest.AsympFit,...
    data.MLP.lag.NodeTest.AsympFit.AIC_min, data.MLP.lag.NodeTest.AsympFit.minNodes] = ...
    AsymptoteFit10(data.MLP.lag, MaxNodes);

% also fit parabola to recognize if #HLN follows u-shaped function.
% avoid selection of minNodes smaller than minimum of parabola by this

[data.MLP.lag.NodeTest.ParabFit.minimum,...
    data.MLP.lag.NodeTest.ParabFit.gof,...
    data.MLP.lag.NodeTest.ParabFit.fitresult] = ...
    fitParabola(data.MLP.lag.NodeTest.AIC, 1:MaxNodes);

if data.MLP.lag.NodeTest.minNodes > data.MLP.lag.NodeTest.AsympFit.minNodes
    if ismember(data.MLP.lag.NodeTest.minNodes,...
            round(data.MLP.lag.NodeTest.ParabFit.minimum) - 1:...
            round(data.MLP.lag.NodeTest.ParabFit.minimum) + 1) &&...
            data.MLP.lag.NodeTest.minNodes < MaxNodes
        data.MLP.lag.NodeTest.AsympFit.flag = false;
        data.MLP.lag.NodeTest.ParabFit.flag = true;
    else
        data.MLP.lag.NodeTest.minNodes = data.MLP.lag.NodeTest.AsympFit.minNodes;
        data.MLP.lag.NodeTest.AIC_min = data.MLP.lag.NodeTest.AIC(data.MLP.lag.NodeTest.minNodes);
        data.MLP.lag.NodeTest.AsympFit.flag = true;
        data.MLP.lag.NodeTest.ParabFit.flag = false;
    end
else
    data.MLP.lag.NodeTest.AsympFit.flag = false;
    data.MLP.lag.NodeTest.ParabFit.flag = false;
end
%% Akaike weights and evidence ratio
data.MLP.lag.NodeTest.deltaI = data.MLP.lag.NodeTest.AIC - data.MLP.lag.NodeTest.AIC_min;
data.MLP.lag.NodeTest.L_est = exp(-0.5.*data.MLP.lag.NodeTest.deltaI);
data.MLP.lag.NodeTest.w_i = data.MLP.lag.NodeTest.L_est ./ sum(data.MLP.lag.NodeTest.L_est);
data.MLP.lag.NodeTest.evidence_ratio = data.MLP.lag.NodeTest.w_i(data.MLP.lag.NodeTest.minNodes)./data.MLP.lag.NodeTest.w_i;

%% sensitivity analysis schmidt 2008 | LAG
BusyText = 'Sensitivity test (Schmidt 2008) 2/3';
jObj.setBusyText(BusyText)

replace_perc = 100;
iterations = 100;
range = .8:.1:1.3;

for k = 1:iterations
    
    [data.sens.lag.RE100.out(:,k),~] = getRE_v4(input.lag, data,replace_perc,'MLP', data.MLP.lag.NodeTest.minNodes); %v4 -> replace with median
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/(iterations*2))*100) '%, 5/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Score.lag.RE100 = zeros(1,length(inHeader.lag));
for k = 1:length(inHeader.lag)
    [counts, ~] = hist(data.sens.lag.RE100.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.lag.RE100(k) = 1;
    end
end
replace_perc = 50;
for k = 1:iterations
    
    [data.sens.lag.RE50.out(:,k),~] = getRE_v4(input.lag, data,replace_perc,'MLP', data.MLP.lag.NodeTest.minNodes); %v4 -> replace with median
    
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str(((iterations+k)/(iterations*2))*100) '%, 6/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Score.lag.RE50 = zeros(1,length(inHeader.lag));
for k = 1:length(inHeader.lag)
    [counts, ~] = hist(data.sens.lag.RE50.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.lag.RE50(k) = 1;
    end
end



%% AIC | BOTH

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding #HLN for MLP with input matrix 3/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

MaxNodes = 20;
iterations = 100;
valchecks = 6;
trainPerc = 70;
valPerc = 30;
testPerc = 0;

for m = 1 : MaxNodes
    
    nodes = m;
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((m/MaxNodes)*100) '%, 7/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    freeParams = (m * size(input.both,1)) + m;
    for k = 1:iterations
        
        [data.MLP.both.output(k,:), data.MLP.both.tr(k,m),data.MLP.both.net{k,m}, ~, ~] =...
            genMLP_v2(input.both,data.ModelIn,nodes,valchecks,trainPerc,valPerc,testPerc);
    end
    
    [data.MLP.both.EnsembleOut(:,m),~,~,data.MLP.both.EnsembleStats.r(m),~,data.MLP.both.EnsembleStats.rmse(m),~] =...
        nNetIterationsMean(data.MLP.both.output', data.ModelIn);
    
    data.MLP.both.NodeTest.AIC(m) =...
        sum(~isnan(data.ModelIn)) * log(data.MLP.both.EnsembleStats.rmse(m)^2) +...
        2*freeParams;
    
    % use bias adjustment term (AICc) if n/k < 40
    
    if (sum(~isnan(data.ModelIn)))/freeParams < 40
        data.MLP.both.NodeTest.AIC(m) = data.MLP.both.NodeTest.AIC(m) +...
            ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
        data.MLP.both.NodeTest.AICc(m) = true;
        
    else
        data.MLP.both.NodeTest.AICc(m) = false;
    end
end

% find minimum AIC
[data.MLP.both.NodeTest.AIC_min, data.MLP.both.NodeTest.minNodes]= min(data.MLP.both.NodeTest.AIC);

% alternative way to find AIC optimum by fitting a 1/x
% function. #HLN is where 1st derivative goes towards zero.

[data.MLP.both.NodeTest.AsympFit,...
    data.MLP.both.NodeTest.AsympFit.AIC_min, data.MLP.both.NodeTest.AsympFit.minNodes] = ...
    AsymptoteFit10(data.MLP.both, MaxNodes);

% also fit parabola to recognize if #HLN follows u-shaped function
% to avoid selection of minNodes smaller than minimum of parabola

[data.MLP.both.NodeTest.ParabFit.minimum,...
    data.MLP.both.NodeTest.ParabFit.gof,...
    data.MLP.both.NodeTest.ParabFit.fitresult] = ...
    fitParabola(data.MLP.both.NodeTest.AIC, 1:MaxNodes);

if data.MLP.both.NodeTest.minNodes > data.MLP.both.NodeTest.AsympFit.minNodes
    if ismember(data.MLP.both.NodeTest.minNodes,...
            round(data.MLP.both.NodeTest.ParabFit.minimum) - 1:...
            round(data.MLP.both.NodeTest.ParabFit.minimum) + 1) &&...
            data.MLP.both.NodeTest.minNodes < MaxNodes
        data.MLP.both.NodeTest.AsympFit.flag = false;
        data.MLP.both.NodeTest.ParabFit.flag = true;
    else
        data.MLP.both.NodeTest.minNodes = data.MLP.both.NodeTest.AsympFit.minNodes;
        data.MLP.both.NodeTest.AIC_min = data.MLP.both.NodeTest.AIC(data.MLP.both.NodeTest.minNodes);
        data.MLP.both.NodeTest.AsympFit.flag = true;
        data.MLP.both.NodeTest.ParabFit.flag = false;
    end
else
    data.MLP.both.NodeTest.AsympFit.flag = false;
    data.MLP.both.NodeTest.ParabFit.flag = false;
end


%% Akaike weights and evidence ratio
data.MLP.both.NodeTest.deltaI = data.MLP.both.NodeTest.AIC - data.MLP.both.NodeTest.AIC_min;
data.MLP.both.NodeTest.L_est = exp(-0.5.*data.MLP.both.NodeTest.deltaI);
data.MLP.both.NodeTest.w_i = data.MLP.both.NodeTest.L_est ./ sum(data.MLP.both.NodeTest.L_est);
data.MLP.both.NodeTest.evidence_ratio = data.MLP.both.NodeTest.w_i(data.MLP.both.NodeTest.minNodes)./data.MLP.both.NodeTest.w_i;
% semilogy(evidence_ratio), grid on, grid minor
%% sensitivity analysis schmidt 2008 | BOTH
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Sensitivity test (Schmidt 2008) 3/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

replace_perc = 100;
iterations = 100;
range = .8:.1:1.3;

for k = 1:iterations
    
    [data.sens.both.RE100.out(:,k),~] = getRE_v4(input.both, data,replace_perc,'MLP', data.MLP.both.NodeTest.minNodes); %v4 -> replace with median
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/(iterations*2))*100) '%, 8/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Score.both.RE100 = zeros(1,length(inHeader.both));
for k = 1:length(inHeader.both)
    [counts, ~] = hist(data.sens.both.RE100.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.both.RE100(k) = 1;
    end
end


replace_perc = 50;


for k = 1:iterations
    
    [data.sens.both.RE50.out(:,k),~] = getRE_v4(input.both, data,replace_perc,'MLP', data.MLP.both.NodeTest.minNodes); %v4 -> replace with median
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str(((iterations+k)/(iterations*2))*100) '%, 9/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Score.both.RE50 = zeros(1,length(inHeader.both));
for k = 1:length(inHeader.both)
    [counts, ~] = hist(data.sens.both.RE50.out(k,:), range);
    MaxCounts = find(max(counts) == counts);
    if MaxCounts == length(range)
        Score.both.RE50(k) = 1;
    end
end

%% garson RI | UNL
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Sensitivity test (Garson 1991) 1/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
iterations = 100; % MLP AIC runs

for k = 1:size(data.MLP.unl.net,1)
    [bias{k}, IW{k}, LW{k}] = separatewb(data.MLP.unl.net{k, data.MLP.unl.NodeTest.minNodes},...
        getwb(data.MLP.unl.net{k,data.MLP.unl.NodeTest.minNodes}));
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/size(data.MLP.unl.net,1))*100) '%, 10/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Garson = plotGarsonhist_v2(IW, LW, inHeader.unl, iterations);

for m = 1:size(inHeader.unl,1)
    for k = 1:iterations
        GarsonRI(k,m) = Garson(k).RI(1,m);
    end
end

[~,MeanInd]=sort(mean(GarsonRI,1), 'descend');
[~,MedInd]=sort(median(GarsonRI,1), 'descend');
[~,MaxInd]=sort(max(GarsonRI), 'descend');

GarsonTable = [inHeader.unl(MeanInd), inHeader.unl(MedInd), inHeader.unl(MaxInd)];
numVars = sum(Score.unl.MLR);
% shorten table to numbers of accepted variables in MLR
GarsonTable = GarsonTable(1:numVars,:);
% find vars that are within the first numVars entries in all three lists
% and assign a score to them
Score.unl.Garson = double(cell2mat(cellfun(@(x) ismember(x,...
    GarsonTable(ismember(GarsonTable(:,3),GarsonTable(ismember(GarsonTable(:,1),GarsonTable(:,2)),1)),3)...
    ), inHeader.unl, 'uniformoutput',0))');
clear GarsonRI
%% garson RI | LAG
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Sensitivity test (Garson 1991) 2/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

for k = 1:size(data.MLP.lag.net,1)
    [bias{k}, IW{k}, LW{k}] = separatewb(data.MLP.lag.net{k, data.MLP.lag.NodeTest.minNodes},...
        getwb(data.MLP.lag.net{k,data.MLP.lag.NodeTest.minNodes}));
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/size(data.MLP.lag.net,1))*100) '%, 11/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Garson = plotGarsonhist_v2(IW, LW, inHeader.lag, iterations);

for m = 1:size(inHeader.lag,1)
    for k = 1:iterations
        GarsonRI(k,m) = Garson(k).RI(1,m);
    end
end

[~,MeanInd]=sort(nanmean(GarsonRI,1), 'descend');
[~,MedInd]=sort(nanmedian(GarsonRI,1), 'descend');
[~,MaxInd]=sort(max(GarsonRI), 'descend');

GarsonTable = [inHeader.lag(MeanInd), inHeader.lag(MedInd), inHeader.lag(MaxInd)];
numVars = sum(Score.lag.MLR);
% shorten table to numbers of accepted variables in MLR
GarsonTable = GarsonTable(1:numVars,:);
% find vars that are within the first numVars entries in all three lists
% and assign a score to them
Score.lag.Garson = double(cell2mat(cellfun(@(x) ismember(x,...
    GarsonTable(ismember(GarsonTable(:,3),GarsonTable(ismember(GarsonTable(:,1),GarsonTable(:,2)),1)),3)...
    ), inHeader.lag, 'uniformoutput',0))');
clear GarsonRI
%% %% garson RI | BOTH
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Sensitivity test (Garson 1991) 3/3';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

for k = 1:size(data.MLP.both.net,1)
    [bias{k}, IW{k}, LW{k}] = separatewb(data.MLP.both.net{k, data.MLP.both.NodeTest.minNodes},...
        getwb(data.MLP.both.net{k,data.MLP.both.NodeTest.minNodes}));
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        '<b>Create scoring table (' num2str((k/size(data.MLP.both.net,1))*100) '%, 12/' num2str(substeps) ')</b><br\>'...
        'Parameterize MLP<br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end

Garson = plotGarsonhist_v2(IW, LW, inHeader.both, iterations);

for m = 1:size(inHeader.both,1)
    for k = 1:iterations
        GarsonRI(k,m) = Garson(k).RI(1,m);
    end
end

[~,MeanInd]=sort(mean(GarsonRI,1), 'descend');
[~,MedInd]=sort(median(GarsonRI,1), 'descend');
[~,MaxInd]=sort(max(GarsonRI), 'descend');

GarsonTable = [inHeader.both(MeanInd), inHeader.both(MedInd), inHeader.both(MaxInd)];
numVars = sum(Score.both.MLR);
% shorten table to numbers of accepted variables in MLR
GarsonTable = GarsonTable(1:numVars,:);
% find vars that are within the first numVars entries in all three lists
% and assign a score to them
Score.both.Garson = double(cell2mat(cellfun(@(x) ismember(x,...
    GarsonTable(ismember(GarsonTable(:,3),GarsonTable(ismember(GarsonTable(:,1),GarsonTable(:,2)),1)),3)...
    ), inHeader.both, 'uniformoutput',0))');

clear bias IW LW Garson GarsonRI


%% calculate level 1 score
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Calculate level 1 score';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

Score.total.unl = sum(cell2mat(struct2cell(Score.unl)));
Score.total.lag = sum(cell2mat(struct2cell(Score.lag)));
Score.total.both = sum(cell2mat(struct2cell(Score.both)));

Score.total.Biomet.all = sum([Score.total.unl(1:size(Biomet.unl,1)), Score.total.lag(1:size(Biomet.lag,1));...
    Score.total.both(1:size(Biomet.both,1))]);
Score.total.Biomet.threshold = ceil(mean(Score.total.Biomet.all(find(Score.total.Biomet.all))));
Score.total.Biomet.best = Score.total.Biomet.all >= Score.total.Biomet.threshold;
Biomet_header.both(Score.total.Biomet.best)
Score.total.FuzzyFP.all = sum([Score.total.unl(size(Biomet.unl,1)+1:end); Score.total.lag(size(Biomet.lag,1)+1:end);...
    Score.total.both(size(Biomet.both,1)+1:end)]);
Score.total.FuzzyFP.threshold = ceil(mean(Score.total.FuzzyFP.all(find(Score.total.FuzzyFP.all))));
Score.total.FuzzyFP.best = Score.total.FuzzyFP.all >= Score.total.FuzzyFP.threshold;
FuzzyFP_header(Score.total.FuzzyFP.best)

input.lvl1 = [Biomet.both(Score.total.Biomet.best,:); FuzzyFP(Score.total.FuzzyFP.best,:)];
inHeader.lvl1 = [Biomet_header.both(Score.total.Biomet.best,:); FuzzyFP_header(Score.total.FuzzyFP.best,:)];


%% MLR filter to gain lvl1.2 input matrix
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Evaluate scores and create level 2 input matrix';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

out = plotStepReg_v4(data.ModelIn, input.lvl1, inHeader.lvl1, timestamp,'');
% out = rmfield(out.MLR, {'unl', 'lag', 'both'});
data.MLR.lvl1 = out; clear out
Score.lvl2.MLR = double(data.MLR.lvl1.inmodel);
data.MLR.lvl1.InSeq = getInSeq(data.MLR.lvl1);
% order inputs according to when the MLR selected them. (use this information to find level 1 MDS inputs: 1st = level1 MDS)
input.lvl2 = input.lvl1(data.MLR.lvl1.InSeq,:);
inHeader.lvl2 = inHeader.lvl1(data.MLR.lvl1.InSeq);
%% remove redundant inputs to gain lvl2 input matrix
% if both an input and its lagged derivatives are in the final list,
% remove the one with the lower Score, remove lagged input if Scores are
% equal


NoBiometVars = length(Score.total.Biomet.all)/2;
% NoBiometVars = length(Biomet.unl);
LagScoreLarger_idx = Score.total.Biomet.all(1:NoBiometVars) < Score.total.Biomet.all(NoBiometVars+1:end);
% index of lagged variables whose score is higher than of the unlagged equivalent

if LagScoreLarger_idx > 0
    LagScoreLarger_idx = strcmp(inHeader.lvl2(cell2mat(cellfun(@find, cellfun(@(x) strcmp(x,inHeader.lvl2),...
        inHeader.lag(LagScoreLarger_idx), 'uniformoutput', 0),'uniformoutput', 0))), inHeader.lvl2);
    match_idx = false(length(inHeader.lvl2),1);
    for k = 1:length(inHeader.lvl2)
        if sum(strcmp([inHeader.lvl2{k} ' #lag#'], inHeader.lvl2(LagScoreLarger_idx)))==1
            match_idx(k)=true;
        end
    end
else
    match_idx = false(length(inHeader.lvl2),1);
end

if sum(match_idx)>0
    inHeader.lvl2 = inHeader.lvl2(~match_idx);
    input.lvl2 = input.lvl2(~match_idx,:);
else
    match_idx = false(length(inHeader.lvl2),1);
    for k = 1:length(inHeader.lvl2)
        match_idx = match_idx + strcmp([inHeader.lvl2{k} ' #lag#'], inHeader.lvl2);
    end
    match_idx = logical(match_idx);
    inHeader.lvl2 = inHeader.lvl2(~match_idx);
    input.lvl2 = input.lvl2(~match_idx,:);
end

%% fit MLR with reduced input space

[data.MLR.lvl2.b,...
    data.MLR.lvl2.bint,...
    data.MLR.lvl2.r,...
    data.MLR.lvl2.rint,...
    data.MLR.lvl2.stats] = regress(data.ModelIn,...
    [input.lvl2' ones(size(input.lvl2',1),1)]); % add a column of ones to obtain intercept


sze = length(data.ModelIn);
data.MLR.lvl2.model = sum(horzcat(repmat(data.MLR.lvl2.b(1:end-1)',sze,1).*input.lvl2' , repmat(data.MLR.lvl2.b(end)', sze,1)),2);
[data.MLR.lvl2.rmse, data.MLR.lvl2.r] = getRhoRMSE(data.ModelIn,data.MLR.lvl2.model);
freeParams = size(input.lvl2,1)*2+1;
data.MLR.lvl2.AIC =...
    sum(~isnan(data.ModelIn)) * log(data.MLR.lvl2.rmse^2) + 2*freeParams;


%% Parameterize MLP with lvl2 input matrix
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding #HLN for MLP...';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||


MaxNodes = 20;
iterations = 100;
valchecks = 6;
trainPerc = 70;
valPerc = 30;
testPerc = 0;

for m = 1 : MaxNodes
    
    nodes = m;
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
        '<b>Parameterize MLP (' num2str((m/MaxNodes)*100) '%)</b><br\>'...
        'Parameterize RBN<br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    freeParams = (m * size(input.lvl2,1)) + m;
    for k = 1:iterations
        
        [data.MLP.lvl2.NodeTest.output(k,:), data.MLP.lvl2.NodeTest.tr(k,m),data.MLP.lvl2.NodeTest.net{k,m}, ~, ~] =...
            genMLP_v2(input.lvl2,data.ModelIn,nodes,valchecks,trainPerc,valPerc,testPerc);
    end
    
    [data.MLP.lvl2.NodeTest.EnsembleOut(:,m),~,~,data.MLP.lvl2.NodeTest.EnsembleStats.r(m),~,data.MLP.lvl2.NodeTest.EnsembleStats.rmse(m),~] =...
        nNetIterationsMean(data.MLP.lvl2.NodeTest.output', data.ModelIn);
    
    data.MLP.lvl2.NodeTest.AIC(m) =...
        sum(~isnan(data.ModelIn)) * log(data.MLP.lvl2.NodeTest.EnsembleStats.rmse(m)^2) +...
        2*freeParams;
    
    % use bias adjustment term (AICc) if n/k < 40
    
    if (sum(~isnan(data.ModelIn)))/freeParams < 40
        data.MLP.lvl2.NodeTest.AIC(m) = data.MLP.lvl2.NodeTest.AIC(m) +...
            ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
        data.MLP.lvl2.NodeTest.AICc(m) = true;
        
    else
        data.MLP.lvl2.NodeTest.AICc(m) = false;
    end
end

% find minimum AIC
[data.MLP.lvl2.NodeTest.AIC_min, data.MLP.lvl2.NodeTest.minNodes]= min(data.MLP.lvl2.NodeTest.AIC);

% alternative way to find AIC optimum by fitting a 1/x
% function. #HLN is where 1st derivative goes towards zero.

[data.MLP.lvl2.NodeTest.AsympFit,...
    data.MLP.lvl2.NodeTest.AsympFit.AIC_min, data.MLP.lvl2.NodeTest.AsympFit.minNodes] = ...
    AsymptoteFit10(data.MLP.lvl2, MaxNodes);

% also fit parabola to recognize if #HLN follows u-shaped function.
% avoid selection of minNodes smaller than minimum of parabola by this

[data.MLP.lvl2.NodeTest.ParabFit.minimum,...
    data.MLP.lvl2.NodeTest.ParabFit.gof,...
    data.MLP.lvl2.NodeTest.ParabFit.fitresult] = ...
    fitParabola(data.MLP.lvl2.NodeTest.AIC, 1:MaxNodes);

if data.MLP.lvl2.NodeTest.minNodes > data.MLP.lvl2.NodeTest.AsympFit.minNodes
    if ismember(data.MLP.lvl2.NodeTest.minNodes,...
            round(data.MLP.lvl2.NodeTest.ParabFit.minimum) - 1:...
            round(data.MLP.lvl2.NodeTest.ParabFit.minimum) + 1) &&...
            data.MLP.lvl2.NodeTest.minNodes < MaxNodes
        data.MLP.lvl2.NodeTest.AsympFit.flag = false;
        data.MLP.lvl2.NodeTest.ParabFit.flag = true;
    else
        data.MLP.lvl2.NodeTest.minNodes = data.MLP.lvl2.NodeTest.AsympFit.minNodes;
        data.MLP.lvl2.NodeTest.AIC_min = data.MLP.lvl2.NodeTest.AIC(data.MLP.lvl2.NodeTest.minNodes);
        data.MLP.lvl2.NodeTest.AsympFit.flag = true;
        data.MLP.lvl2.NodeTest.ParabFit.flag = false;
    end
else
    data.MLP.lvl2.NodeTest.AsympFit.flag = false;
    data.MLP.lvl2.NodeTest.ParabFit.flag = false;
end


%% Akaike weights and evidence ratio
data.MLP.lvl2.NodeTest.deltaI = data.MLP.lvl2.NodeTest.AIC - data.MLP.lvl2.NodeTest.AIC_min;
data.MLP.lvl2.NodeTest.L_est = exp(-0.5.*data.MLP.lvl2.NodeTest.deltaI);
data.MLP.lvl2.NodeTest.w_i = data.MLP.lvl2.NodeTest.L_est ./ sum(data.MLP.lvl2.NodeTest.L_est);
data.MLP.lvl2.NodeTest.evidence_ratio = data.MLP.lvl2.NodeTest.w_i(data.MLP.lvl2.NodeTest.minNodes)./data.MLP.lvl2.NodeTest.w_i;

%% Parameterize RBN with lvl2 input matrix

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding #HLN for RBN...';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

searchRange = [1 3 5 7 10:10:70];
iterations = 100;

for m = searchRange
    
    nodes = m;
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
        'Parameterize MLP (' num2str(100) '%)<br\>'...
        '<b>Parameterize RBN (' num2str((find(searchRange==m)/length(searchRange))*100) '%)</b><br\>'...
        'Run models'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
    % run RBN
    [data.RBN.Net, data.RBN.Pred, data.RBN.RMSE, data.RBN.Rho, data.RBN.Spread] =...
        genRBN_v3(input.lvl2,data.ModelIn,iterations,nodes);
    
    [~,~,~,data.RBN.NodeTest.r(m),~,data.RBN.NodeTest.rmse(m),~] = ...
        nNetIterationsMean(data.RBN.Pred,data.ModelIn);
    
    freeParams = data.RBN.Net{1}.numWeightElements;
    
    data.RBN.NodeTest.AIC(m) =...
        sum(~isnan(data.ModelIn)) * log(data.RBN.NodeTest.rmse(m)^2) +...
        2*freeParams;
    
end


%% run MLP with lvl 2 inputs
substeps_2 = 6;
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'MLP...';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

iterations = 1000;
valchecks = 6;
trainPerc = 70;
valPerc = 30;
testPerc = 0;
nodes = data.MLP.lvl2.NodeTest.minNodes;
freeParams = size(input.lvl2,1)*nodes + nodes + nodes + 1;

for k = 1:iterations
    
    [data.MLP.lvl2.output(k,:), data.MLP.lvl2.tr(k),data.MLP.lvl2.net{k}, ~, ~] =...
        genMLP_v2(input.lvl2,data.ModelIn,nodes,valchecks,trainPerc,valPerc,testPerc);
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
        'Parameterize MLP (' num2str(100) '%)<br\>'...
        'Parameterize RBN (' num2str(100) '%)<br\>'...
        '<b>Run models (' num2str((k/iterations)*100) '%, 1/' num2str(substeps_2) ')</b>'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
end


[data.MLP.lvl2.EnsembleOut,~,~,data.MLP.lvl2.EnsembleStats.r,~,data.MLP.lvl2.EnsembleStats.rmse,~] =...
    nNetIterationsMean(data.MLP.lvl2.output', data.ModelIn);


data.MLP.lvl2.AIC =...
    sum(~isnan(data.ModelIn)) * log(data.MLP.lvl2.EnsembleStats.rmse^2) +...
    2*freeParams;

% use bias adjustment term (AICc) if n/k < 40

if (sum(~isnan(data.ModelIn)))/freeParams < 40
    data.MLP.lvl2.AIC = data.MLP.lvl2.AIC +...
        ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
    data.MLP.lvl2.AICc = true;
else
    data.MLP.lvl2.AICc = false;
end

%% run RBN with lvl 2 inputs
iterations = 100;
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'RBN...';
jObj.setBusyText(BusyText)
List = ['Create input matrices (' num2str(100) '%)<br\>'...
    'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
    'Parameterize MLP (' num2str(100) '%)<br\>'...
    'Parameterize RBN (' num2str(100) '%)<br\>'...
    '<b>Run models (2/' num2str(substeps_2) ')</b>'];
updateProgessWindow(List)
drawnow
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

[~,idx] = min(data.RBN.NodeTest.AIC(searchRange));
data.RBN.NodeTest.minNodes = searchRange(idx);

[data.RBN.Net, data.RBN.Pred, data.RBN.rmse, data.RBN.r, data.RBN.spread] =...
    genRBN_v3(input.lvl2,data.ModelIn,iterations,data.RBN.NodeTest.minNodes);

[data.RBN.CM.pred,~,~,data.RBN.CM.r,~,data.RBN.CM.rmse,~] = ...
    nNetIterationsMean(data.RBN.Pred,data.ModelIn);

freeParams = data.RBN.Net{1}.numWeightElements;
data.RBN.CM.AIC =...
    sum(~isnan(data.ModelIn)) * log(data.RBN.CM.rmse^2) +...
    2*freeParams;

% use bias adjustment term (AICc) if n/k < 40

if (sum(~isnan(data.ModelIn)))/freeParams < 40
    data.RBN.CM.AIC = data.RBN.CM.AIC +...
        ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
    data.RBN.CM.AICc = true;
else
    data.RBN.CM.AICc = false;
end

%% run GRN with lvl 2 inputs
iterations = 100;
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'GRNN...';
jObj.setBusyText(BusyText)
List = ['Create input matrices (' num2str(100) '%)<br\>'...
    'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
    'Parameterize MLP (' num2str(100) '%)<br\>'...
    'Parameterize RBN (' num2str(100) '%)<br\>'...
    '<b>Run models (3/' num2str(substeps_2) ')</b>'];
updateProgessWindow(List)
drawnow
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

[data.GRN.Net, data.GRN.pred, data.GRN.rmse, data.GRN.r, data.GRN.spread] = genGRN(input.lvl2, data.ModelIn, iterations,.6);
data.GRN.CM.pred = nanmean(data.GRN.pred,2);
[data.GRN.CM.rmse, data.GRN.CM.r] = getRhoRMSE(data.ModelIn, data.GRN.CM.pred);
data.GRN.CM.AIC = sum(~isnan(data.ModelIn)) * log(data.GRN.CM.rmse^2) + 2*data.GRN.Net{1}.numWeightElements;

% use bias adjustment term (AICc) if n/k < 40

if (sum(~isnan(data.ModelIn)))/freeParams < 40
    data.GRN.CM.AIC = data.GRN.CM.AIC +...
        ((2*freeParams*(freeParams+1))/(sum(~isnan(data.ModelIn))-freeParams-1));
    data.GRN.CM.AICc = true;
else
    data.GRN.CM.AICc = false;
end


%%  MDS Limits
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding MDS initial limits...';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

clear rmse
iterations = 1000;
n_min = 5;

% create 5000 random limits
% use (max-min)/2 as variation interval for Lim.rand
Lim.rand.lvl1 = ((max(input.lvl2(1,:)) - min(input.lvl2(1,:)))/2)*rand(iterations,1);

for k = 2:size(input.lvl2,1)
    
    Lim.rand.lvl2(:,k-1) = ((max(input.lvl2(k,:)) - min(input.lvl2(k,:)))/2)*rand(iterations,1);
    
end

for k = 1:iterations
    [data.MDS.out, ~,data.MDS.SolQ]=...
        Reichstein_gpfll_QC_v6(data.ModelIn, data.err,...
        timestamp, n_min,...
        input.lvl2(1,:),...
        Lim.rand.lvl1(k),...
        input.lvl2(2:end,:),...
        Lim.rand.lvl2(k,:));
    
    [rmse(k),rho(k),~] = getRhoRMSE(data.ModelIn,data.MDS.out);
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
        'Parameterize MLP (' num2str(100) '%)<br\>'...
        'Parameterize RBN (' num2str(100) '%)<br\>'...
        '<b>Run models (' num2str((k/iterations)*100) '%, 4/' num2str(substeps_2) ')</b>'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
end

% find best (lowest mse) combination of random input limits to use as
% initial limits in next step
bins = 5;
Best = 100;

[Lim.init.lvl1] = findInitLim(bins, Best, Lim.rand.lvl1, rmse.^2);
[Lim.init.lvl2] = findInitLim(bins, Best, Lim.rand.lvl2, rmse.^2);
clear rmse

%
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'Finding MDS limits...';
jObj.setBusyText(BusyText)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

iterations = 1000;
n_min = 5;

% create 1000 random limits
% use (max-min)/2 as variation interval for Lim.rand
Lim.rand2.lvl1 = ((max(input.lvl2(1,:)) - min(input.lvl2(1,:)))/2)*rand(iterations,1);

for k = 2:size(input.lvl2,1)
    
    Lim.rand2.lvl2(:,k-1) = ((max(input.lvl2(k,:)) - min(input.lvl2(k,:)))/2)*rand(iterations,1);
    
end


for k = 1:iterations
    [data.MDS.out, ~,data.MDS.SolQ]=...
        Reichstein_gpfll_QC_v6(data.ModelIn, data.err,...
        timestamp, n_min,...
        input.lvl2(1,:),...
        Lim.rand2.lvl1(k),...
        input.lvl2(2:end,:),...
        Lim.init.lvl2);
    [Lim.init.stats.rmse(k),Lim.init.stats.rho(k),~] = getRhoRMSE(data.ModelIn,data.MDS.out);
end

[Lim.best.lvl1] = findInitLim(bins, Best, Lim.rand2.lvl1, Lim.init.stats.rmse.^2);
clear rmse
bins = 5;
Best = 100;

LimVector = nan(iterations, size(Lim.rand2.lvl2,2));
for m = 1:length(Lim.init.lvl2)
    for k = 1:iterations
        idx = ~ismember(1:length(Lim.init.lvl2),m);
        LimVector(k,idx) = Lim.init.lvl2(idx);
        LimVector(k,m) = Lim.rand2.lvl2(k,m);
        
        [data.MDS.out, ~,data.MDS.SolQ]=...
            Reichstein_gpfll_QC_v6(data.ModelIn, data.err,...
            timestamp, n_min,...
            input.lvl2(1,:),...
            Lim.init.lvl1,...
            input.lvl2(2:end,:),...
            LimVector(k,:));
        
        [Lim.best.stats.rmse(k,m),Lim.best.stats.rho(k,m),~] = getRhoRMSE(data.ModelIn,data.MDS.out);
    end
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    List = ['Create input matrices (' num2str(100) '%)<br\>'...
        'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
        'Parameterize MLP (' num2str(100) '%)<br\>'...
        'Parameterize RBN (' num2str(100) '%)<br\>'...
        '<b>Run models (' num2str((m/length(Lim.init.lvl2))*100) '%, 5/' num2str(substeps_2) ')</b>'];
    updateProgessWindow(List)
    drawnow
    % ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
    
end

[Lim.best.lvl2] = findInitLim(bins, Best, Lim.rand2.lvl2, Lim.best.stats.rmse.^2);

% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||
BusyText = 'MDS...';
jObj.setBusyText(BusyText)
List = ['Create input matrices (' num2str(100) '%)<br\>'...
    'Create scoring table (' num2str(100) '%, 12/' num2str(substeps) ')<br\>'...
    'Parameterize MLP (' num2str(100) '%)<br\>'...
    'Parameterize RBN (' num2str(100) '%)<br\>'...
    '<b>Run models (6/' num2str(substeps_2) ')</b>'];
updateProgessWindow(List)
drawnow
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

[data.MDS.out,~,~,data.MDS.SolQ]=...
    Reichstein_gpfll_v5(data.ModelIn, data.err,...
    timestamp, n_min,...
    input.lvl2(1,:),...
    Lim.best.lvl1,...
    input.lvl2(2:end,:),...
    Lim.best.lvl2);

[data.MDS.QC.out,~,data.MDS.QC.SolQ]=...
    Reichstein_gpfll_QC_v6(data.ModelIn, data.err,...
    timestamp, n_min,...
    input.lvl2(1,:),...
    Lim.best.lvl1,...
    input.lvl2(2:end,:),...
    Lim.best.lvl2);


[data.MDS.rmse, data.MDS.rho] = getRhoRMSE(data.MDS.QC.out, data.ModelIn);
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||

jObj.setBusyText('Done!');
jObj.stop;
close(Prog_wndw)
% ||||||||||||||||| Progress Window |||||||||||||||||||||||||||||||||||||||


%%%% END OF MAIN PROGRAM

    function [ maxCov,LagTime,lagged_time,cross_cov_series] = lagtime_30min_v3(Var2, Var1, days, color)
        
        % including normalized cross-correlation(Kettunen et al. 1996), not only cross-covariance
        
        % ||||||||| Var1 -> gets shifted about timelag,
        % ||||||||| Var2 -> reference var
        
        meanVar1 = nanmean(Var1);
        meanVar2 = nanmean(Var2);
        % Cov_Var1_Var2 = mean((Var1-meanVar1).*(Var2-meanVar2));
        % AvgVar1 = mean(Var1);
        % StdVar1 = std(Var1);
        % days = 2;
        
        time_steps = -days * 24 * 2 : 1 : days * 24 * 2;
        
        cross_cov_series = NaN(size(time_steps));
        cross_corr_series = NaN(size(time_steps));
        for i=1:length(time_steps)
            if time_steps(i)<0
                Var1_in = Var1(1:end+time_steps(i));
                Var2_in = Var2(-time_steps(i)+1:end);
            elseif time_steps(i)>0
                Var1_in = Var1(time_steps(i)+1:end);
                Var2_in = Var2(1:end-time_steps(i));
            else
                Var1_in = Var1;
                Var2_in = Var2;
            end
            cross_cov_series(i) = nanmean((Var1_in - meanVar1).*(Var2_in - meanVar2)); %calculate the covariance for this lag time, put it into the series
            cross_corr_series(i) = nansum((Var1_in - meanVar1) .* (Var2_in - meanVar2))/...
                sqrt(nansum((Var1_in - meanVar1).^2) * nansum((Var1_in - meanVar1).^2));
        end
        
        lagged_time = time_steps / 2; %calculate the lag time as a time (units: hours) rather than as time steps
        
        % simple lag-determination method:
        % this is EdiRe's way of finding peak - subtract mean of endpoints
        avgCovRange = mean(cross_cov_series(:,[1 end]),2);
        [~,indMax] = max(abs(cross_cov_series - avgCovRange),[],2);
        LagTime = lagged_time(indMax);
        maxCov = cross_cov_series(indMax);
        
        % plot(lagged_time, mapminmax(cross_corr_series), 'color', color)
        % % hold on, plot(LagTime, maxCov, 'r+')
        % xlabel('Timelag, hrs')
        
        % T = sqrt(sum(~isnan(Var2))-2) .* ((mapminmax(cross_corr_series))./sqrt(1-(mapminmax(cross_corr_series)).^2));
        
        
    end

    function [AsympFit, AIC_min, minNodes] = AsymptoteFit100(input, MaxNodes)
        %% CO2 version
        % minNodes definition: 1st derivative is smaller than 100,
        % when rounded to the 1st digit to the left of the decimal point
        
        [AsympFit.result,...
            AsympFit.gof,...
            AsympFit.asymptote,...
            AsympFit.inflect_pt,...
            AsympFit.f] =...
            fit1byx_v2(input.NodeTest.AIC');
        
        
        
        
        % %% view fit result
        % close all
        % subplot(3,1,1)
        % plot(input.NodeTest.AIC'), hold on
        % plot(input.NodeTest.AsympFit.result)
        % set(gca, 'xlim', [1 MaxNodes])
        % subplot(3,1,2)
        % % plot(input.NodeTest.AIC'), hold on
        % % ezplot(diff(input.NodeTest.AsympFit.f))
        % plot(subs(diff(input.NodeTest.AsympFit.f),1:1:MaxNodes))
        % set(gca, 'xlim', [1 MaxNodes])
        % subplot(3,1,3)
        % % plot(input.NodeTest.AIC'), hold on
        % plot(subs(diff(diff(input.NodeTest.AsympFit.f)), 1:1:MaxNodes))
        % set(gca, 'xlim', [1 MaxNodes])
        
        %% find best #HLN
        % best #HLN is where derivative is smaller than abs(100) for the first time.
        % round 1st derivative of fitted "1/x" to two digits to the left of the decimal point
        % definition: Best #HLN is where 1st derivative is zero for the first time
        
        % overwrite *.minNodes and *.AIC_min
        minNodes = MaxNodes - sum(iszero(round(double(subs(diff(AsympFit.f),1:1:MaxNodes)),-2))) + 1;
        AIC_min = input.NodeTest.AIC(input.NodeTest.minNodes);
        
        function [fitresult, gof, asympt, inflec_pt,f] = fit1byx_v2(data)
            %% fit f = a/x + b to data
            % get asymptote & inflection point
            
            ft = fittype('a/x+b');
            
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            
            [fitresult, gof] = fit((1:length(data))',data, ft, opts );
            
            coeffs = coeffvalues(fitresult);
            x = sym('x');
%             x = x; % avoids static workspace error since fuction is nested
            f = coeffs(1)/x + coeffs(2);
            asympt = double(limit(f, inf)); % asymptote
            f2 = diff(diff(f)); % second derivative of f
            inflec_pt = solve(f2 == 0);
            % inflec_pt = 1;
        end
        
        
        %% view best #HLN determination results
        % close all
        % plot(double(subs(input.NodeTest.AsympFit.f,1:1:MaxNodes))), hold on
        % plot(input.NodeTest.AIC(1:MaxNodes)')
        % ylimits = get(gca,'ylim');
        % plot(input.NodeTest.BestNodes,...
        %     input.NodeTest.AIC(input.NodeTest.BestNodes),'r+',...
        %     'Markersize', 20, 'linewidth', 2)
        % set(gca, 'xlim', [0 MaxNodes])
        % lgnd = legend('');
        % set(lgnd, 'visible', 'off')
        % title('dra, unl')
        % xlabel('#HLN')
        % ylabel('AIC')
        % text(20, ylimits(1)+(ylimits(2)-ylimits(1))/2,...
        %     ['r^2 of fit: ' num2str(input.NodeTest.AsympFit.gof.rsquare)])
        
        
        
    end

    function [AsympFit, AIC_min, minNodes] = AsymptoteFit10(input, MaxNodes)
        %% Methane version.
        % minNodes definition: 1st derivative is smaller than 10,
        % when rounded to the 1st digit to the left of the decimal point
        
        [AsympFit.result,...
            AsympFit.gof,...
            AsympFit.asymptote,...
            AsympFit.inflect_pt,...
            AsympFit.f] =...
            fit1byx_v2(input.NodeTest.AIC');
        
        
        % %% view fit result
        % close all
        % subplot(3,1,1)
        % plot(input.NodeTest.AIC'), hold on
        % plot(input.NodeTest.AsympFit.result)
        % set(gca, 'xlim', [1 MaxNodes])
        % subplot(3,1,2)
        % % plot(input.NodeTest.AIC'), hold on
        % % ezplot(diff(input.NodeTest.AsympFit.f))
        % plot(subs(diff(input.NodeTest.AsympFit.f),1:1:MaxNodes))
        % set(gca, 'xlim', [1 MaxNodes])
        % subplot(3,1,3)
        % % plot(input.NodeTest.AIC'), hold on
        % plot(subs(diff(diff(input.NodeTest.AsympFit.f)), 1:1:MaxNodes))
        % set(gca, 'xlim', [1 MaxNodes])
        
        %% find best #HLN
        % best #HLN is where derivative is smaller than abs(10) for the first time.
        % round 1st derivative of fitted "1/x" to one digit to the left of the decimal point
        % definition: Best #HLN is where 1st derivative is zero for the first time
        
        % overwrite *.minNodes and *.AIC_min
        minNodes = MaxNodes - sum(iszero(round(double(subs(diff(AsympFit.f),1:1:MaxNodes)),-1))) + 1;
        AIC_min = input.NodeTest.AIC(input.NodeTest.minNodes);
        
        function [fitresult, gof, asympt, inflec_pt,f] = fit1byx_v2(data)
            %% fit f = a/x + b to data
            % get asymptote & inflection point
            
            ft = fittype('a/x+b');
            
            opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            opts.Display = 'Off';
            
            [fitresult, gof] = fit((1:length(data))',data, ft, opts );
            
            coeffs = coeffvalues(fitresult);
            x = sym('x');
            f = coeffs(1)/x + coeffs(2);
            asympt = double(limit(f, inf)); % asymptote
            f2 = diff(diff(f)); % second derivative of f
            inflec_pt = solve(f2 == 0);
            % inflec_pt = 1;
        end
        %% view best #HLN determination results
        % close all
        % plot(double(subs(input.NodeTest.AsympFit.f,1:1:MaxNodes))), hold on
        % plot(input.NodeTest.AIC(1:MaxNodes)')
        % ylimits = get(gca,'ylim');
        % plot(input.NodeTest.BestNodes,...
        %     input.NodeTest.AIC(input.NodeTest.BestNodes),'r+',...
        %     'Markersize', 20, 'linewidth', 2)
        % set(gca, 'xlim', [0 MaxNodes])
        % lgnd = legend('');
        % set(lgnd, 'visible', 'off')
        % title('dra, unl')
        % xlabel('#HLN')
        % ylabel('AIC')
        % text(20, ylimits(1)+(ylimits(2)-ylimits(1))/2,...
        %     ['r^2 of fit: ' num2str(input.NodeTest.AsympFit.gof.rsquare)])
        
        
        
    end

    function [minimum, gof, fitresult] = fitParabola(in, nodes)
        
        
        % in = data.MLP.NodeTest.AIC';
        
        [fitresult, gof] = fit(nodes',in', 'poly2');
        coeffs = coeffvalues(fitresult);
        x = sym('x');
        f = sym('f');
        f = coeffs(1)*x.^2 + coeffs(2).*x + coeffs(3);
        f1 = diff(f);
        minimum = double(solve(f1==0));
        
        
    end

    function [InitLim] = findInitLim(bins, Best, randLim, mse)
        
        for k = 1:size(randLim,2)
            MinIdx = GetMinIdx(mse,Best);
            [counts, center] = hist(randLim(MinIdx,k),bins);
            [~,maxpos] = max(counts);
            InitLim(k) = center(maxpos);
        end
    end

    function [rbNet, pred, rmse, rho, spread] = genGRN(input, target,iterations, trainperc)
        
        %%
        rbIn = normalize(input')';
        rbIn(isnan(rbIn)) = 0; %replace NaNs (CC) with zeros
        rbTar = (target)';
        nanIdx = isnan(rbTar); % remove NaNs
        rbIn(:,nanIdx) = [];
        rbTar(nanIdx)=[];
        % iterations = 100;
        
        rbNet = cell(length(iterations),1);
        rmse = zeros(length(iterations),1);
        rho = zeros(length(iterations),1);
        spread = zeros(length(iterations),1);
        % test_pred = zeros(length(iterations),1);
        pred = zeros(length(target),length(iterations));
        % wb = waitbar(0,'...');
        
        for k = 1:iterations
            
            %     waitbar(k/iterations,wb)
            trainIdx = randperm(length(rbTar),floor(length(rbTar)*trainperc));
            testIdx = 1:length(rbTar);
            testIdx(trainIdx)=[];
            test_spread = .2 : .1 : 1.5;
            rbNet_spread = cell(length(test_spread),1);
            mse_spread = zeros(length(test_spread),1);
            rho_spread = zeros(length(test_spread),1);
            
            for m = 1 : length(test_spread)
                rbNet_spread{m} = newgrnn(rbIn(:,trainIdx), rbTar(trainIdx), test_spread(m));
                pred_spread = sim(rbNet_spread{m},rbIn(:,testIdx) );
                [mse_spread(m), rho_spread(m)] = getRhoMSE(rbTar(testIdx)', pred_spread');
            end
            
            [~,spread_idx] = min(mse_spread);
            spread(k) = test_spread(spread_idx);
            
            rbNet{k} = newgrnn(rbIn(:,trainIdx), rbTar(trainIdx), spread(k));
            test_pred = sim(rbNet{k},rbIn(:,testIdx));
            [rmse(k), rho(k)] = getRhoRMSE(rbTar(testIdx)', test_pred');
            pred(:,k) = sim(rbNet{k},normalize(input')'); %simulate RBN over all inputs
            
        end
        % close(wb)
        
        
    end

    function [output,tr, net, TrainInput, TrainTarget] = genMLP_v2(input, target, nodes,valchecks,trainPerc,valPerc,testPerc)
        
        TrainTarget = target';
        TrainInput = input;
        TrainInput(:,isnan(target))=[];
        TrainTarget(isnan(target))=[];
        
        % Create a Fitting Network
        % hiddenLayerSize = 20;
        net = feedforwardnet(nodes);
        
        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = trainPerc/100;
        net.divideParam.valRatio = valPerc/100;
        net.divideParam.testRatio = testPerc/100;
        net.trainParam.max_fail = valchecks;
        net.trainFcn = 'trainlm';
        net.trainParam.showWindow = 0;
        
        
        % Train the Network
        % tic
        % [net,tr] = train(net,TrainInput,TrainTarget,'useParallel','yes');
        [net,tr] = train(net,TrainInput,TrainTarget);
        
        
        % Simulate the Network over whole input space
        % output = net(input,'useParallel','yes');
        output = net(input);
        % toc
        
        % output = output';
        % target = target';
        % [r, p]=corr(output(isnan(output)==0 &...
        % isnan(target)==0) ,target(isnan(output)==0 &...
        % isnan(target)==0),'type','pearson');
        % dev = target-output;
        % rmse=sqrt(nansum(dev.^2)/(sum(~isnan(dev))));
        %
        % weightbias=getwb(net);
        %
        % [bias,IW,LW] = separatewb(net,weightbias);
        
    end

    function [rbNet, pred, rmse, rho, spread] = genRBN_v3(input, target,iterations, nodes)
        
        %%
        rbIn = normalize(input')';
        % rbIn(isnan(rbIn)) = 0; %replace NaNs (CC) with zeros
        rbTar = (target)';
        nanIdx = isnan(rbTar); % remove NaNs
        rbIn(:,nanIdx) = [];
        rbTar(nanIdx)=[];
        nanIdx = find(sum(isnan(rbIn)));
        rbIn(:,nanIdx) = [];
        rbTar(nanIdx)=[];
        
        % iterations = 100;
        
        rbNet = cell(length(iterations),1);
        rmse = zeros(length(iterations),1);
        rho = zeros(length(iterations),1);
        spread = zeros(length(iterations),1);
        % test_pred = zeros(length(iterations),1);
        pred = zeros(length(target),length(iterations));
        % wb = waitbar(0,'...');
        
        for k = 1:iterations
            
            %     waitbar(k/iterations,wb)
            trainIdx = randperm(length(rbTar),floor(length(rbTar)*.6));
            testIdx = 1:length(rbTar);
            testIdx(trainIdx)=[];
            test_spread = .2 : .1 : 2;
            rbNet_spread = cell(length(test_spread),1);
            mse_spread = zeros(length(test_spread),1);
            rho_spread = zeros(length(test_spread),1);
            
            for m = 1 : length(test_spread)
                rbNet_spread{m} = newrbEDIT(rbIn(:,trainIdx), rbTar(trainIdx), .1, test_spread(m), nodes, nodes); %.1 is error goal, low because
                pred_spread = rbNet_spread{m}(rbIn(:,testIdx));                                                   % not supposed to be reached
                [mse_spread(m), rho_spread(m)] = getRhoMSE(rbTar(testIdx)', pred_spread');
            end
            
            [~,spread_idx] = min(mse_spread);
            spread(k) = test_spread(spread_idx);
            rbNet{k} = newrbEDIT(rbIn(:,trainIdx), rbTar(trainIdx), .1, spread(k), nodes, nodes);
            test_pred = rbNet{k}(rbIn(:,testIdx));
            [rmse(k), rho(k)] = getRhoRMSE(rbTar(testIdx)', test_pred');
            pred(:,k) = rbNet{k}(normalize(input')'); %simulate RBN over all inputs
            
        end
        % close(wb)
        
        function [out1,out2] = newrbEDIT(varargin)
            %NEWRB Design a radial basis network. || Console output supressed in this
            %version !
            %
            %  Radial basis networks can be used to approximate functions.  <a href="matlab:doc newrb">newrb</a>
            %  adds neurons to the hidden layer of a radial basis network until it
            %  meets the specified mean squared error goal.
            %
            %  <a href="matlab:doc newrb">newrb</a>(X,T,GOAL,SPREAD,MN,DF) takes these arguments,
            %    X      - RxQ matrix of Q input vectors.
            %    T      - SxQ matrix of Q target class vectors.
            %    GOAL   - Mean squared error goal, default = 0.0.
            %    SPREAD - Spread of radial basis functions, default = 1.0.
            %    MN     - Maximum number of neurons, default is Q.
            %    DF     - Number of neurons to add between displays, default = 25.
            %  and returns a new radial basis network.
            %
            %  The larger that SPREAD is the smoother the function approximation
            %  will be.  Too large a spread means a lot of neurons will be
            %  required to fit a fast changing function.  Too small a spread
            %  means many neurons will be required to fit a smooth function,
            %  and the network may not generalize well.  Call NEWRB with
            %  different spreads to find the best value for a given problem.
            %
            %  Here we design a radial basis network given inputs X and targets T.
            %
            %    X = [1 2 3];
            %    T = [2.0 4.1 5.9];
            %    net = <a href="matlab:doc newrb">newrb</a>(X,T);
            %    Y = net(X)
            %
            %  See also SIM, NEWRBE, NEWGRNN, NEWPNN.
            
            % Mark Beale, 11-31-97
            % Copyright 1992-2011 The MathWorks, Inc.
            % $Revision: 1.1.6.19 $ $Date: 2013/10/15 05:52:50 $
            
            %% =======================================================
            %  BOILERPLATE_START
            %  This code is the same for all Network Functions.
            
            persistent INFO;
            if isempty(INFO), INFO = get_info; end
            if (nargin > 0) && ischar(varargin{1}) ...
                    && ~strcmpi(varargin{1},'hardlim') && ~strcmpi(varargin{1},'hardlims')
                code = varargin{1};
                switch code
                    case 'info',
                        out1 = INFO;
                    case 'check_param'
                        err = check_param(varargin{2});
                        if ~isempty(err), nnerr.throw('Args',err); end
                        out1 = err;
                    case 'create'
                        if nargin < 2, error(message('nnet:Args:NotEnough')); end
                        param = varargin{2};
                        err = nntest.param(INFO.parameters,param);
                        if ~isempty(err), nnerr.throw('Args',err); end
                        out1 = create_network(param);
                        out1.name = INFO.name;
                    otherwise,
                        % Quick info field access
                        try
                            out1 = eval(['INFO.' code]);
                        catch %#ok<CTCH>
                            nnerr.throw(['Unrecognized argument: ''' code ''''])
                        end
                end
            else
                [args,param] = nnparam.extract_param(varargin,INFO.defaultParam);
                [param,err] = INFO.overrideStructure(param,args);
                if ~isempty(err), nnerr.throw('Args',err,'Parameters'); end
                [net,tr] = create_network(param);
                net.name = INFO.name;
                out1 = net;
                out2 = tr;
            end
        end
        
        function v = fcnversion
            v = 7;
        end
        
        %  BOILERPLATE_END
        %% =======================================================
        
        function info = get_info
            info = nnfcnNetwork(mfilename,'Radial Basis Network',fcnversion, ...
                [ ...
                nnetParamInfo('inputs','Input Data','nntype.data',{0},...
                'Input data.'), ...
                nnetParamInfo('targets','Target Data','nntype.data',{0},...
                'Target output data.'), ...
                nnetParamInfo('goal','Performance Goal','nntype.pos_scalar',0,...
                'Performance goal.'), ...
                nnetParamInfo('spread','Radial basis spread','nntype.strict_pos_scalar',1,...
                'Distance from radial basis center to 0.5 output.'), ...
                nnetParamInfo('maxNeurons','Maximum number of neurons','nntype.pos_int_inf_scalar',inf,...
                'Maximum number of neurons to add to network.'), ...
                nnetParamInfo('displayFreq','Display Frequency','nntype.strict_pos_int_scalar',50,...
                'Number of added neurons between displaying progress at command line.'), ...
                ]);
        end
        
        function err = check_param(param)
            err = '';
        end
        
        function [net,tr] = create_network(param)
            
            % Data
            p = param.inputs;
            t = param.targets;
            if iscell(p), p = cell2mat(p); end
            if iscell(t), t = cell2mat(t); end
            
            % Max Neurons
            Q = size(p,2);
            mn = param.maxNeurons;
            if (mn > Q), mn = Q; end
            
            
            % Dimensions
            R = size(p,1);
            S2 = size(t,1);
            
            % Architecture
            net = network(1,2,[1;1],[1; 0],[0 0;1 0],[0 1]);
            
            % Simulation
            net.inputs{1}.size = R;
            net.layers{1}.size = 0;
            net.inputWeights{1,1}.weightFcn = 'dist';
            net.layers{1}.netInputFcn = 'netprod';
            net.layers{1}.transferFcn = 'radbas';
            net.layers{2}.size = S2;
            net.outputs{2}.exampleOutput = t;
            
            % Performance
            net.performFcn = 'mse';
            
            % Design Weights and Bias Values
            warn1 = warning('off','MATLAB:rankDeficientMatrix');
            warn2 = warning('off','MATLAB:nearlySingularMatrix');
            [w1,b1,w2,b2,tr] = designrb(p,t,param.goal,param.spread,mn,param.displayFreq);
            warning(warn1.state,warn1.identifier);
            warning(warn2.state,warn2.identifier);
            
            net.layers{1}.size = length(b1);
            net.b{1} = b1;
            net.iw{1,1} = w1;
            net.b{2} = b2;
            net.lw{2,1} = w2;
        end
        
        %======================================================
        function [w1,b1,w2,b2,tr] = designrb(p,t,eg,sp,mn,df)
            
            [r,q] = size(p);
            [s2,q] = size(t);
            b = sqrt(-log(.5))/sp;
            
            % RADIAL BASIS LAYER OUTPUTS
            P = radbas(dist(p',p)*b);
            PP = sum(P.*P)';
            d = t';
            dd = sum(d.*d)';
            
            % CALCULATE "ERRORS" ASSOCIATED WITH VECTORS
            e = ((P' * d)' .^ 2) ./ (dd * PP');
            
            % PICK VECTOR WITH MOST "ERROR"
            pick = findLargeColumn(e);
            used = [];
            left = 1:q;
            W = P(:,pick);
            P(:,pick) = []; PP(pick,:) = [];
            e(:,pick) = [];
            used = [used left(pick)];
            left(pick) = [];
            
            % CALCULATE ACTUAL ERROR
            w1 = p(:,used)';
            a1 = radbas(dist(w1,p)*b);
            [w2,b2] = solvelin2(a1,t);
            a2 = w2*a1 + b2*ones(1,q);
            MSE = mse(t-a2);
            
            % Start
            tr = newtr(mn,'perf');
            tr.perf(1) = mse(t-repmat(mean(t,2),1,q));
            tr.perf(2) = MSE;
            %   if isfinite(df)
            %     fprintf('NEWRB, neurons = 0, MSE = %g\n',tr.perf(1));
            %   end
            flag_stop=plotperfrb(tr,eg,'NEWRB',0);
            
            iterations = min(mn,q);
            for k = 2:iterations
                
                % CALCULATE "ERRORS" ASSOCIATED WITH VECTORS
                wj = W(:,k-1);
                a = wj' * P / (wj'*wj);
                P = P - wj * a;
                PP = sum(P.*P)';
                e = ((P' * d)' .^ 2) ./ (dd * PP');
                
                % PICK VECTOR WITH MOST "ERROR"
                pick = findLargeColumn(e);
                W = [W, P(:,pick)];
                P(:,pick) = []; PP(pick,:) = [];
                e(:,pick) = [];
                used = [used left(pick)];
                left(pick) = [];
                
                % CALCULATE ACTUAL ERROR
                w1 = p(:,used)';
                a1 = radbas(dist(w1,p)*b);
                [w2,b2] = solvelin2(a1,t);
                a2 = w2*a1 + b2*ones(1,q);
                MSE = mse(t-a2);
                
                % PROGRESS
                tr.perf(k+1) = MSE;
                
                %     % DISPLAY
                %     if isfinite(df) & (~rem(k,df))
                %       fprintf('NEWRB, neurons = %g, MSE = %g\n',k,MSE);
                %       flag_stop=plotperfrb(tr,eg,'NEWRB',k);
                %     end
                
                % CHECK ERROR
                if (MSE < eg), break, end
                if (flag_stop), break, end
                
            end
            
            [S1,R] = size(w1);
            b1 = ones(S1,1)*b;
            
            % Finish
            if isempty(k), k = 1; end
            tr = cliptr(tr,k);
        end
        
        %======================================================
        
        function i = findLargeColumn(m)
            replace = find(isnan(m));
            m(replace) = zeros(size(replace));
            m = sum(m .^ 2,1);
            i = find(m == max(m));
            i = i(1);
        end
        
        %======================================================
        
        function [w,b] = solvelin2(p,t)
            if nargout <= 1
                w= t/p;
            else
                [pr,pc] = size(p);
                x = t/[p; ones(1,pc)];
                w = x(:,1:pr);
                b = x(:,pr+1);
            end
        end
        
        %======================================================
        
        function stop=plotperfrb(tr,goal,name,epoch)
            
            % Error check: must be at least one argument
            if nargin < 1, error(message('nnet:Args:NotEnough')); end
            
            % NNT 5.1 Backward compatibility
            if (nargin == 1) && ischar(tr)
                stop = 1;
                return
            end
            
            % Defaults
            if nargin < 2, goal = NaN; end
            if nargin < 3, name = 'Training Record'; end
            if nargin < 4, epoch = length(tr.epoch)-1; end
            
            % Special case 2: Delete plot if zero epochs
            if (epoch == 0) || isnan(tr.perf(1))
                fig = find_existing_figure;
                if ~isempty(fig), delete(fig); end
                if (nargout), stop = 0; end
                return
            end
            
            % Special case 3: No plot if performance is NaN
            if (epoch == 0) || isnan(tr.perf(1))
                if (nargout) stop = 0; end
                return
            end
            
            % GET FIGURE AND USER DATA
            % ========================
            
            % Get existing/new figure
            fig2 = find_existing_figure;
            if isempty(fig2), fig2 = new_figure(name); end
            figure(fig2);
            
            % Get existing/new userdata
            ud=get(fig2,'userdata');
            if isempty(ud)
                createNewPlot(fig2);
                ud = get(fig2,'userdata');
            end
            
            % UPDATE PLOTTING DATA
            % ====================
            
            % Epoch indices and initial y-limits
            ind = 1:(epoch+1);
            ymax=1e-20;
            ymin=1e20;
            
            % Update performance plot and ylimits
            set(ud.TrainLine(2),...
                'Xdata',tr.epoch(ind),...
                'Ydata',tr.perf(ind),...
                'linewidth',2,'color','b');
            ymax=(max([ymax tr.perf(ind)]));
            ymin=(min([ymin tr.perf(ind)]));
            
            % Update performance goal plot and y-limits (if required)
            % plot goal only if > 0, or if 0 and ymin is also 0
            plotGoal = isfinite(goal) & ((goal > 0) | (ymin == 0));
            if plotGoal
                set(ud.TrainLine(1),...
                    'Xdata',tr.epoch(ind),...
                    'Ydata',goal+zeros(1,epoch+1),...
                    'linewidth',2,'color','k');
                ymax=(max([ymax goal]));
                ymin=(min([ymin goal]));
            end
            
            % Update axis scale and rounded y-limits
            if (ymin > 0)
                yscale = 'log';
                ymax=10^ceil(log10(ymax));
                ymin=10^fix(log10(ymin)-1);
            else
                yscale = 'linear';
                ymax=10^ceil(log10(ymax));
                ymin=0;
            end
            set(ud.TrainAxes,'xlim',[0 epoch],'ylim',[ymin ymax]);
            set(ud.TrainAxes,'yscale',yscale);
            
            % UPDATE FIGURE TITLE, NAME, AND AXIS LABLES
            % ====================
            
            % Update figure title
            tstring = sprintf('Performance is %g',tr.perf(epoch+1));
            if isfinite(goal)
                tstring = [tstring ', ' sprintf('Goal is %g',goal)];
            end
            set(ud.TrainTitle,'string',tstring);
            
            % Update figure name
            if isempty(name)
                set(fig2,'name',['Training with ' upper(tr.trainFcn)],'numbertitle','off');
            end
            
            % Update axis x-label
            if epoch == 0
                set(ud.TrainXlabel,'string','Zero Epochs');
            elseif epoch == 1
                set(ud.TrainXlabel,'string','One Epoch');
            else
                set(ud.TrainXlabel,'string',[num2str(epoch) ' Epochs']);
            end
            
            % Update axis y-lable
            set(ud.TrainYlabel,'string','Performance');
            
            % FINISH
            % ======
            
            % Make changes now
            drawnow;
            
            % Return stop flag if required
            if (nargout), stop = 0; end
        end
        
        %======================================================
        
        % Find pre-existing figure, if any
        function fig = find_existing_figure
            % Initially assume figure does not exist
            fig = [];
            % Search children of root...
            for child=get(0,'children')'
                % ...for objects whose type is figure...
                if strcmp(get(child,'type'),'figure')
                    % ...whose tag is 'train'
                    if strcmp(get(child,'tag'),'train')
                        % ...and stop search if found.
                        fig = child;
                        break
                    end
                end
            end
            % Not sure if/why this is necessary
            if isempty(get(fig,'children'))
                fig = [];
            end
        end
        
        %======================================================
        
        % New figure
        function fig = new_figure(name)
            fig = figure(...
                'Units',          'pixel',...
                'Name',           name,...
                'Tag',            'train',...
                'NumberTitle',    'off',...
                'IntegerHandle',  'off',...
                'Toolbar',        'none');
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Create new plot in figure
        
        function createNewPlot(fig)
            % Delete all children from figure
            z = get(fig,'children');
            for i=1:length(z)
                delete (z(i));
            end
            
            % Create axis
            ud.TrainAxes     = axes('Parent',fig);
            ud.TrainLine     = plot(0,0,0,0,0,0,0,0,'Parent',ud.TrainAxes);
            ud.TrainXlabel   = xlabel('X Axis','Parent',ud.TrainAxes);
            ud.TrainYlabel   = ylabel('Y Axis','Parent',ud.TrainAxes);
            ud.TrainTitle    = get(ud.TrainAxes,'Title');
            set(ud.TrainAxes,'yscale','log');
            ud.XData      = [];
            ud.YData      = [];
            ud.Y2Data     = [];
            set(fig,'UserData',ud,'menubar','none','toolbar','none');
            
            legend(ud.TrainLine(2),'Train');
            
            % Bring figure to front
            figure(fig);
        end
        
        function tr=newtr(epochs,varargin)
            names = varargin;
            tr.epoch = 0:epochs;
            blank = zeros(1,epochs+1)+NaN;
            for i=1:length(names)
                eval(['tr.' names{i} '=blank;']);
            end
        end
        
        %======================================================
        
        function tr=cliptr(tr,epochs)
            indices = 1:(epochs+1);
            names = fieldnames(tr);
            for i=1:length(names)
                name = names{i};
                value = tr.(name);
                if isnumeric(value) && (numel(value) > epochs)
                    tr.(name) = value(:,indices);
                end
            end
        end
        
        %======================================================
        
        
        
    end

    function [in_idx] = getInSeq(in)
        
        in_idx = nan(size(in.history.in,1),1);
        
        for k=1:size(in.history.in,1)
            
            switch (k)
                case 1
                    in_idx(k) = find(in.history.in(k,:));
                otherwise
                    prev_idx =  find(in.history.in(k-1,:));
                    new_idx = find(in.history.in(k,:));
                    if length(prev_idx)  > length(new_idx)
                        remove = setdiff(prev_idx, new_idx);
                    else
                        in_idx(k) = setdiff(new_idx,prev_idx);
                    end
            end
        end
        
        if exist('remove', 'var')==1
            in_idx(in_idx==remove)= [];
        end
        in_idx(isnan(in_idx))= [];
    end

    function [RE, flux] = getRE_v4(input, flux, replace_perc, networktype,varargin)
        % replace 'replace_perc' % of in var with average and observe effect on MSE.
        % median statt mean!
        
        flux.sens.r = zeros(size(input,1), 1);
        flux.sens.rmse = zeros(size(input,1), 1);
        
        rowNo = size(input,2);
        %     nodes = 5;
        valchecks = 10;
        trainPerc = 50;
        valPerc = 40;
        testPerc = 10;
        if ~isempty(varargin) && strcmp(networktype,'RBN')==1
            RBNnodes = varargin{1};
        elseif ~isempty(varargin) && strcmp(networktype,'MLP')==1
            MLPnodes = varargin{1};
        end
        
        if strcmp(networktype,'MLP')==1
            [flux.mlPred,~,flux.ref.Net,~,~] =...
                genMLP_v2(input,flux.ModelIn,MLPnodes,valchecks,trainPerc,valPerc,testPerc);
            flux.ref.rmse = getRhoRMSE(flux.ModelIn, flux.mlPred');
        elseif strcmp(networktype,'RBN')==1
            [flux.rbNet, flux.rbPred, flux.ref.rmse, flux.rbRho, flux.rbSpread] =...
                genRBN_v3(input,flux.ModelIn,1,RBNnodes);
            %         flux.rbPred = nanmean(flux.rbPred,2);
            flux.ref.rmse = getRhoRMSE(flux.ModelIn, flux.rbPred);
        elseif strcmp(networktype, 'GRN')==1
            [flux.grNet, flux.grPred, flux.ref.rmse, flux.grRho, flux.grSpread] = genGRN(input, flux.ModelIn,1);
            %         flux.grPred = nanmean(flux.grPred,2);
            flux.ref.rmse = getRhoRMSE(flux.ModelIn, flux.grPred);
        end
        %%
        
        % size(input) runs with one artificial var each
        flux.sens.model = nan(rowNo, size(input,1));
        
        for m = 1 : size(input,1)
            NumIdx = find(~isnan(flux.ModelIn));
            sze = length(NumIdx);
            replace_RowIdx = randperm(sze, floor(sze * (replace_perc/100)));
            replace_idx = NumIdx(replace_RowIdx);
            input_art = input;
            input_art(m,replace_idx) = nanmedian(input(m,:));
            if strcmp(networktype,'MLP')==1
                flux.sens.model(:,m) = flux.ref.Net(input_art)'; % simulate model with artificial input vars
            elseif strcmp(networktype,'RBN')==1
                flux.sens.model(:,m) = flux.rbNet{1}(normalize(input_art')'); % simulate model with artificial input vars
            elseif strcmp(networktype, 'GRN')==1
                flux.sens.model(:,m) = flux.grNet{1}(normalize(input_art')');
            end
            flux.sens.rmse(m) = getRhoRMSE(flux.ModelIn, flux.sens.model(:,m));
            
        end
        
        
        RE = (flux.sens.rmse.^2)/(flux.ref.rmse^2);
    end

    function [rmse,rho,p] = getRhoRMSE(data,model)
        %GETRHORMSE Summary of this function goes here
        %   Detailed explanation goes here
        
        dev = data - model;
        rmse = sqrt(nansum(dev.^2) / (sum(~isnan(dev))));
        
        [rho, p] = corr(model(isnan(model)==0 &...
            isnan(data)==0) ,data(isnan(model)==0 &...
            isnan(data)==0),'type','pearson');
    end

    function [mse,rho,p] = getRhoMSE(data,model)
        %GETRHORMSE Summary of this function goes here
        %   Detailed explanation goes here
        
        dev = data - model;
        mse = nansum(dev.^2) / (sum(~isnan(dev)));
        
        [rho, p] = corr(model(isnan(model)==0 &...
            isnan(data)==0) ,data(isnan(model)==0 &...
            isnan(data)==0),'type','pearson');
    end

    function [MEAN,MEDIAN,MAD,r,p,rmse,dev] = nNetIterationsMean( all, measured )
        %NNETITERATIONSMEAN Summary of this function goes here
        %   Detailed explanation goes here
        
        MEAN=mean(all,2);
        MEDIAN=median(all,2);
        % MAD=mad(all);
        MAD=nanmedian(abs(measured-MEAN));
        
        
        [r, p]=corr(MEAN(isnan(measured)==0 &...
            isnan(MEAN)==0) ,measured(isnan(measured)==0 &...
            isnan(MEAN)==0),'type','pearson');
        dev=measured-MEAN;
        rmse=sqrt(nansum(dev.^2)/(sum(~isnan(dev))));
        
        
        
    end

    function [Garson] = plotGarsonhist_v2(IW, LW, header, iterations)
        nr_input_vars = size(header,1);
        Garson = struct('c',cell(iterations,1),'r',cell(iterations,1),'S',cell(iterations,1),'RI',cell(iterations,1));
        
        % ch4
        
        for n = 1: iterations
            Garson(n).c = IW{n}{1} .* repmat((LW{n}{2})',1,nr_input_vars);
            Garson(n).r = abs(Garson(n).c)./repmat(sum(abs(Garson(n).c),2),1,nr_input_vars);
            Garson(n).S = sum(Garson(n).r,1);
            Garson(n).RI = Garson(n).S / sum(Garson(n).S);
        end
        
        %     clear min
        
        %     figure('position',[1 1 1680 970], 'visible', visible)
        
        for m = 1 : nr_input_vars
            
            subplot(ceil(nr_input_vars/6),6,m)
            RI = nan(iterations,1);
            
            for n = 1 : iterations
                %         plot(k,Garson(k).RI(m),'+')
                %         hist(k,Garson(k).RI(m))
                ylim([0 .3])
                RI(n) = Garson(n).RI(m);
                %         hold on,
            end
            %         [freq,center]=hist(RI,0:.05:.3);
            %         bar(center,freq,'facecolor','k','edgecolor','w')
            %         set(gca,'xlim', [0 .4], 'ylim', [0 iterations])
            %         %     hist(RI)
            %         text(.1,iterations-iterations*.2,{char(['median: ' num2str(median(RI))]);...
            %             char(['mean: '  num2str(mean(RI))]);...
            %             char(['max: '  num2str(max(RI))]);...
            %             char(['min: '  num2str(min(RI))]);...
            %             }),% 'backgroundcolor', [1 1 1])
            %         title(header(m))
            %
            %
            %
            %         textaxes = axes('position',[0 0 1 1],'visible','off');
            %         set(gcf, 'currentaxes', textaxes)
            %         text(.3,.98, char(['Relative importance of neural net inputs after Garson ' fluxname]),'fontsize',15, 'fontweight', 'bold')
        end
    end

    function [MLR] = plotStepReg_v4(target, input, inHeader, timestamp, caption)
        
        sze = length(target);
        
        
        [MLR.b,...
            MLR.se,...
            MLR.pval,...
            MLR.inmodel,...
            MLR.stats,...
            MLR.nextstep,...
            MLR.history] = stepwisefit(input',target);
        
        
        MLR.model = sum(horzcat(repmat(MLR.inmodel .* MLR.b',sze,1) .* input',...
            repmat(MLR.stats.intercept,sze,1)),2);
        [MLR.rmse, MLR.r] = getRhoRMSE(target,MLR.model);
        freeParams = size(input,1)*2+1;
        MLR.AIC =...
            sum(~isnan(target)) * log(MLR.rmse^2) + 2*freeParams;
        
        
        % NULL = 0;
        % NULL = timeseries(NULL, timestamp);
        % NULL = formatTimeseries(NULL);
        %
        % figure('position', [1 1 1680 970])
        %
        %
        %
        % plot(NULL, 'linestyle', 'none'), hold on,
        % plot(getTimevector(NULL), target, 'color', [.6 .6 .6], 'linestyle', 'none', 'marker', '+'), hold on
        % plot(getTimevector(NULL), MLR.model, 'color', 'g')
        % % ylabel('F_{CH_4}')
        % ylabel('')
        % XTicks = get(gca,'xtick');
        %
        % text(XTicks(end-4), 100, char(['Sequence of used input vars:';inHeader(getInSeq(MLR))]))
        % text(XTicks(end-2), 100, char(['Rejected input vars:';inHeader(~MLR.inmodel)]))
        % text(XTicks(end-6), 120, char({['r: ' num2str(MLR.r)];...
        %     ['RMSE: ' num2str(MLR.rmse)];...
        %     ['data coverage: ' num2str(100-(sum(isnan(MLR.model))/...
        %     length(MLR.model))*100) ' %'];...
        %     ['AIC: ' num2str(MLR.AIC)]}))
        %
        % title(caption)
    end

    function [] = updateProgessWindow(List)
        
        % dim = [200 400];
        ListObj = com.mathworks.widgets.HyperlinkTextLabel(List);
        jPanel = ListObj.getHTMLPane;
        color = get(gcf, 'color');
        ListObj.setBackgroundColor(java.awt.Color(color(1), color(2), color(3)));
        javacomponent(jPanel, [1 100 300 100], gcf);
        drawnow
        
    end

    function [idx] = GetMinIdx(mse, lngth)
        
        [~, idx] = sort(mse, 'ascend');
        idx = idx(1:lngth);
        
        
        %     r_sorted = sort(r,'descend');
        %     [~,idx,~] = intersect(r,r_sorted);
        %     idx = idx(end-lgth+1:end);
        %     idx = sort(idx,'descend');
        
    end


end


