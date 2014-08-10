function cv_mlc_perf_linear(datafile,loss_type)

addpath('dataset_cv');
addpath('qpc');
addpath('utils');

% datafile = 'emotions_5';
% loss_type = 'example_fone';

%     'hamming',...
%     'example_fone','example_precision','example_recall', 'example_preck','example_reck','example_prbep','example_accuracy',...
%     'macro_fone','macro_precision','macro_recall', 'macro_preck','macro_reck','macro_prbep','macro_accuracy',...
%     'micro_fone','micro_precision','micro_recall', 'micro_preck','micro_reck','micro_prbep','micro_accuracy'


save_file = sprintf('results/%s_mlc_perf_%s_1',datafile,loss_type);

load(datafile);
% X Y nfold test_fold_idx train_fold_idx

% setting parameters
opt.loss_type = loss_type;
opt.maxiter = 3000;
opt.eps = 0.01; %0.01 0.5;
opt.threshold = 1e-8; % reduce cut epsilon
opt.k = 5;
opt.display = 0;

Cset = [0.01 0.1 1 10 100];

csize = length(Cset);
scores = cell(csize,1);
times = zeros(csize,nfold);

for k=1:nfold
    tr_idx = train_fold_idx{k};
    te_idx = test_fold_idx{k};
    
    trX = X(tr_idx,:);
    trY = Y(tr_idx,:)';
    teX = X(te_idx,:);
    teY = Y(te_idx,:)';
    
    [L,N] = size(trY);
  
    for c=1:csize
        C = Cset(c);
        
        fprintf('k=%d,C=%f...\n',k,C);
        
        level = get_level(loss_type);
        switch(level)
            case 1
                tempC = N * L * C;
            case 2
                tempC = N * C;
            case 3
                tempC = L * C;
            case 4
                tempC = C;
            otherwise
                error('level is not defined\n');
        end
        
        t = cputime;
        W = mlc_perf_linear_sum_sparse(trY,trX,tempC,opt);
        times(c,k) = cputime - t;
        
        predteYscore = full(W' * teX'); 
        [score,measure_list] = eval_all_measures(teY, predteYscore,opt.k);
        for i=1:length(measure_list)
            fprintf('%s : %f\n',measure_list{i},score(i));
        end
        
        scores{c} =[scores{c}, score];  
    end       
end

save(save_file, 'scores','times');

