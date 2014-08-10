function [mvY, b, xi] = find_most_violated_config(Y, predYscore,loss_type,k)

% loss_type
% Hamming loss            'hamming'
% Fone loss               'fone' 
% Precision loss          'precision'
% Precision at k loss     'preck'
% Recall loss             'recall'
% Recall at k loss        'reck'
% PRBEP                   'prbep' 
% Accuracy                'accuracy'  

% Example_based measure by adding prefix example_

% Macro_based measure by adding prefix marco_

% Micro_based measure by adding prefix micro_

% k is the ratio of training data

if nargin == 3
    k = 0;
elseif nargin < 3 || nargin >4
    error(' parameter: y, the score of predy, losstype,k');
end

switch (loss_type)
    case 'hamming'
        [mvY, b, xi] = find_most_violated_hamming_loss(Y, predYscore);
        
    case 'example_fone'        
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,1,k); % fone: loss_function=1
    case 'example_precision'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,6,k); % precision: loss_function=6
    case 'example_recall'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,7,k); % recall: loss_function=7
    case 'example_preck'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,4,k); % preck: loss_function=4
    case 'example_reck'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,5,k); % reck: loss_function=5
    case 'example_prbep'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,3,k); % prbep: loss_function=3
    case 'example_accuracy'
        [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,8,k); % accuracy: loss_function=8
        
    case 'macro_fone'        
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,1,k); % fone: loss_function=1
    case 'macro_precision'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,6,k); % precision: loss_function=6
    case 'macro_recall'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,7,k); % recall: loss_function=7
    case 'macro_preck'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,4,k); % preck: loss_function=4
    case 'macro_reck'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,5,k); % reck: loss_function=5
    case 'macro_prbep'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,3,k); % prbep: loss_function=3
    case 'macro_accuracy'
        [mvY, b, xi] = find_most_violated_macro_loss(Y, predYscore,8,k); % accuracy: loss_function=8
        
    case 'micro_fone'        
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,1,k); % fone: loss_function=1
    case 'micro_precision'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,6,k); % precision: loss_function=6
    case 'micro_recall'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,7,k); % recall: loss_function=7
    case 'micro_preck'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,4,k); % preck: loss_function=4
    case 'micro_reck'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,5,k); % reck: loss_function=5
    case 'micro_prbep'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,3,k); % prbep: loss_function=3
    case 'micro_accuracy'
        [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,8,k); % accuracy: loss_function=8             
        
    otherwise
        error('the loss is not defined\n');
end

% the objective function is
% max_Y  F(X,Y;W) = 1/(N * L) sum_i sum_l w_l^T \phi(x_i)

% predYscore(l,i) = w_l^T \phi(x_i) 
function [mvY, b, xi] = find_most_violated_hamming_loss(Y, predYscore)

[L,N] = size(Y);
mvY = Y;
xi = 0;
b = 0;
for i=1:N
    err = 1 - 2 * Y(:,i) .* predYscore(:,i);
    idx = find(err > 0);
    mvY(idx,i) = -1 .* mvY(idx,i); % err>0, then reverse the label
    xi = xi + sum(err(idx));
    b = b + length(idx);
end
xi = xi / (N * L);
b = b / (N * L);

% predYscore(l,i) = 1/L w_l^T \phi(x_i)
function [mvY, b, xi] = find_most_violated_example_loss(Y, predYscore,loss_function,k)
[L,N] = size(Y);
mvY = zeros(L,N);
xi = 0;
b = 0;

for i=1:N
    [tempy,tempb,tempxi] = thresholdmetric(Y(:,i),predYscore(:,i),loss_function,k);
    mvY(:,i) = tempy;
    xi = xi + tempxi - Y(:,i)' * predYscore(:,i);
    b = b + tempb;
end

xi = xi / N;
b = b / N;

% predYscore(l,i) = 1/N w_l^T \phi(x_i)
function [mvY, b, xi] = find_most_violated_macro_loss(Y,predYscore,loss_function,k)

[L,N] = size(Y);
mvY = zeros(L,N);
xi = 0;
b = 0;

for j=1:L
   [tempy,tempb,tempxi] = thresholdmetric(Y(j,:)',predYscore(j,:)',loss_function,k); 
   mvY(j,:) = tempy';
   xi = xi + tempxi - Y(j,:) * predYscore(j,:)';
   b = b + tempb;
end

xi = xi / L;
b = b / L;

% predYscore(l,i) = 1/(N*L) w_l^T \phi(x_i)
function [mvY, b, xi] = find_most_violated_micro_loss(Y, predYscore,loss_function,k)
[L,N] = size(Y);

temp_Y = reshape(Y,L*N,1);
temp_predYscore = reshape(predYscore,L*N,1);

[tempy,b,xi] =thresholdmetric(temp_Y, temp_predYscore,loss_function,k);

mvY = reshape(tempy,L,N);
xi = xi - temp_Y' *temp_predYscore;
