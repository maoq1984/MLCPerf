function level = get_level(loss_type)

if strcmp(loss_type,'hamming')
    level = 1;
elseif strcmp(loss_type,'example_fone') || strcmp(loss_type,'example_precision')||...
        strcmp(loss_type,'example_recall') || strcmp(loss_type,'example_preck')||...
        strcmp(loss_type,'example_reck') || strcmp(loss_type,'example_prbep')||...
        strcmp(loss_type,'example_accuracy')
    level = 2;
elseif strcmp(loss_type,'macro_fone') || strcmp(loss_type,'macro_precision')||...
        strcmp(loss_type,'macro_recall') || strcmp(loss_type,'macro_preck')||...
        strcmp(loss_type,'macro_reck') || strcmp(loss_type,'macro_prbep')||...
        strcmp(loss_type,'macro_accuracy')
    level = 3;
elseif strcmp(loss_type,'micro_fone') || strcmp(loss_type,'micro_precision')||...
        strcmp(loss_type,'micro_recall') || strcmp(loss_type,'micro_preck')||...
        strcmp(loss_type,'micro_reck') || strcmp(loss_type,'micro_prbep')||...
        strcmp(loss_type,'micro_accuracy')
    level = 4;
else
    error('loss_type is not defined\n');
end
    