function [scores,measure_list] = eval_all_measures(Y, predYscore,k)

measure_list = {'hamming',...
     'example_fone','example_precision','example_recall', 'example_preck','example_reck','example_prbep','example_accuracy',...
     'macro_fone','macro_precision','macro_recall', 'macro_preck','macro_reck','macro_prbep','macro_accuracy',...
     'micro_fone','micro_precision','micro_recall', 'micro_preck','micro_reck','micro_prbep','micro_accuracy'};
 
nsize = length(measure_list);
scores = zeros(nsize,1);
 
for i=1:nsize
    
    if strcmp(measure_list{i},'example_preck') || strcmp(measure_list{i},'example_reck') ||...
            strcmp(measure_list{i},'example_preck') || strcmp(measure_list{i},'example_reck') ||...
            strcmp(measure_list{i},'example_preck') || strcmp(measure_list{i},'example_reck')        
        scores(i) = eval_performance(Y,predYscore,measure_list{i},k);
    else
        scores(i) = eval_performance(Y,predYscore,measure_list{i});
    end
    
end