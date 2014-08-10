function score = eval_performance(Y, predYscore,loss_type,k)

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
        score = eval_hamming(Y,predYscore);
 
    case 'example_fone'        
        score = eval_example_measure(Y,predYscore,1,k); % fone: loss_function=1
    case 'example_precision'
        score = eval_example_measure(Y,predYscore,6,k); % precision: loss_function=6
    case 'example_recall'
        score = eval_example_measure(Y,predYscore,7,k); % recall: loss_function=7
    case 'example_preck'
        score = eval_example_measure(Y,predYscore,4,k); % preck: loss_function=4
    case 'example_reck'
        score = eval_example_measure(Y,predYscore,5,k); % reck: loss_function=5
    case 'example_prbep'
        score = eval_example_measure(Y,predYscore,3,k); % prbep: loss_function=3
    case 'example_accuracy'
        score = eval_example_measure(Y,predYscore,8,k); % accuracy: loss_function=8
       
    case 'macro_fone'   
        score = eval_macro_measure(Y,predYscore,1,k); % fone: loss_function=1
    case 'macro_precision'
        score = eval_macro_measure(Y,predYscore,6,k); % precision: loss_function=6
    case 'macro_recall'
        score = eval_macro_measure(Y,predYscore,7,k); % recall: loss_function=7
    case 'macro_preck'
        score = eval_macro_measure(Y,predYscore,4,k); % preck: loss_function=4
    case 'macro_reck'
        score = eval_macro_measure(Y,predYscore,5,k); % reck: loss_function=5
    case 'macro_prbep'
        score = eval_macro_measure(Y,predYscore,3,k); % prbep: loss_function=3
    case 'macro_accuracy'
        score = eval_macro_measure(Y,predYscore,8,k); % accuracy: loss_function=8
        
    case 'micro_fone'    
        score = eval_micro_measure(Y,predYscore,1,k); % fone: loss_function=1
    case 'micro_precision'
        score = eval_micro_measure(Y,predYscore,6,k); % precision: loss_function=6
    case 'micro_recall'
        score = eval_micro_measure(Y,predYscore,7,k); % recall: loss_function=7
    case 'micro_preck'
        score = eval_micro_measure(Y,predYscore,4,k); % preck: loss_function=4
    case 'micro_reck'
        score = eval_micro_measure(Y,predYscore,5,k); % reck: loss_function=5
    case 'micro_prbep'
        score = eval_micro_measure(Y,predYscore,3,k); % prbep: loss_function=3
    case 'micro_accuracy'
        score = eval_micro_measure(Y,predYscore,8,k); % accuracy: loss_function=8     
 
    otherwise
        error('the loss is not defined\n');
end

function score = eval_hamming(Y,predYscore)
[L,N] = size(predYscore);
score = sum(sum(sign(predYscore) ~= Y)) / (L * N);

function score = eval_example_measure(Y,predYscore,loss_type,k)
N = size(predYscore,2);

score = 0;
for i=1:N    
    if k==0
        score = score + eval_prediction(Y(:,i),predYscore(:,i),loss_type);
    else
        tmp_predYscore = -1 .* ones(size(predYscore(:,i)));
        [val,idx] = sort(predYscore(:,i),'descend'); % choose the best k to be labeled as 1
        tmp_predYscore(idx(1:k)) = 1;
        score = score + eval_prediction(Y(:,i),tmp_predYscore,loss_type);
    end
end
score = score/N;

function score = eval_macro_measure(Y,predYscore,loss_type,k)
L = size(predYscore,1);

score = 0;
for j=1:L
    if k==0
        score = score + eval_prediction(Y(j,:)',predYscore(j,:)',loss_type);
    else
        tmp_predYscore = -1 .* ones(size( predYscore(j,:)' ));
        [val,idx] = sort(predYscore(j,:),'descend');
        tmp_predYscore(idx(1:k)) = 1;
        score = score + eval_prediction(Y(j,:)',tmp_predYscore,loss_type);
    end 
end
score = score/L;

function score = eval_micro_measure(Y,predYscore,loss_type,k)
[L,N] = size(predYscore);

tempY = reshape(Y,L*N,1);
tempPredY = reshape(predYscore,L*N,1);
if k==0
    score = eval_prediction(tempY,tempPredY,loss_type);
else
    tmp_predYscore = -1 .* ones(size(tempPredY));
    [val,idx] = sort(tempPredY,'descent');
    tmp_predYscore(idx(1:k)) = 1;
    score = eval_prediction(tempY,tmp_predYscore,loss_type);
end