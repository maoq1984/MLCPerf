function [W,history] = mlc_perf_linear_sum_sparse(Y,X,C,opt)

loss_type = opt.loss_type;

eps = opt.eps;
maxiter = opt.maxiter;
k = opt.k; % prec@k, recall@k
cut_remove_threshold = opt.threshold;

[L,N] = size(Y);
[N1,D] = size(X);

if(N ~= N1)
    error(' the size of X and Y mismatch\n');
end

% H matrix
Q = zeros(maxiter,maxiter);
bv = zeros(maxiter,1);

% most violated constratin set 
MVY = sparse([]);

% initalization
n_active = 0;
iter = 0;
W = zeros(D,L);
history.time = cputime;

gamma = get_gamma(loss_type,N,L);

predYscore = W' * X';
[mvY,b,xi] = find_most_violated_config(Y, predYscore,loss_type,k);

gap_array = [xi];
history.upper_bound_array = [];
history.low_bound_array = [];

while 1
    
    % add the most violated constraint
    deltaY = Y - mvY;
    MVY = [MVY,reshape(deltaY,L*N,1)];
    n_active = n_active+1;
    iter = iter + 1;
    
    % update the Q matrix and bv
    tmp_k = reshape(deltaY * X, D*L,1);
    for i=1:n_active
        tmp_i = reshape(reshape(MVY(:,i),L,N) * X, D*L, 1);
        Q(n_active,i) = tmp_k' * tmp_i / gamma^2;
        if(n_active == i)
            Q(n_active,i) = Q(n_active,i) + 1e-10; % avoid singular value
        else
           Q(i,n_active) = Q(n_active,i); % symmetric matrix
        end        
    end
    
    bv(n_active) = b;
    
    % solve QP
    f = -1 .* bv(1:n_active);
    cL = ones(1,n_active);
    ck = C;
    xl = zeros(n_active,1);
    xu = inf .* ones(n_active,1);
    H = Q(1:n_active,1:n_active);    
        
    alpha= qpas(H,f,cL,ck,[],[],xl,xu,0);
    
        % reduce cuts
    nz = find(alpha >= cut_remove_threshold);
    tmp_n_active = length(nz);
    if(tmp_n_active ~= n_active && tmp_n_active ~= 0 ) % reduce 
        temp_Q = Q(nz,nz);
        Q = zeros(maxiter,maxiter);
        Q(1:tmp_n_active,1:tmp_n_active) = temp_Q;
        
        temp_bv = bv(nz);
        bv = zeros(maxiter,1);
        bv(1:tmp_n_active) = temp_bv;
        
        MVY = MVY(:,nz);
        
        n_active = tmp_n_active;
        
        % retrain
        f = -1 .* bv(1:n_active);
        cL = ones(1,n_active);
        ck = C;
        xl = zeros(n_active,1);
        xu = inf .* ones(n_active,1);
        H = Q(1:n_active,1:n_active);    

        alpha= qpas(H,f,cL,ck,[],[],xl,xu,0);
    end
    
    
    % calculate solution
    weight_mvy = repmat(alpha',L*N,1) .* MVY(:,1:n_active);
    beta = reshape(sum(weight_mvy,2),L,N);
    W = (beta * X)' ./ gamma;
    
    % for different level, we need to calculate different val
    predYscore = full(W' * X');
        
    % finding most violated constraint
    [mvY, b, xi] = find_most_violated_config(Y,predYscore,loss_type,k);
        
    % calculate the lower bound
    low_bounds = zeros(n_active,1);
    tmp = reshape(predYscore,L*N,1);
    for i=1:n_active
        tmp_i = reshape(reshape(MVY(:,i),L,N), L*N,1);
        low_bounds = bv(i) - (tmp' * tmp_i)/gamma;
    end
    max_low_bound = max([low_bounds;0]);
    
    %% terminate conditions
    gap = xi - max_low_bound; 
    
    if opt.display
        fprintf('iter:%d, No cut: %d upper_bound: %f, lower_bound:%f, gap:%f \n'...
             ,iter,n_active, xi, max_low_bound,gap);
    else
        if (mod(iter+1,100) == 0)
            fprintf('.');
        end
    end
    
    % terminate when dual gap do not change 
    gap_array = [gap_array;gap];
    history.upper_bound_array = [history.upper_bound_array,xi];
    history.low_bound_array = [history.low_bound_array,max_low_bound];
    
    gap_array_size = length(gap_array);
    if(gap_array_size > 10)
        gap_array = gap_array((gap_array_size-10):gap_array_size);
    end
    diff_vals = diff(gap_array);
    max_diff = max(abs(diff_vals));    
    
    if(max_low_bound + eps >= xi || iter >= maxiter || max_diff < 0.001 * eps)        
        if(max_diff < 0.001 * eps)
            fprintf('gap variation no change.\n')
        end
        if(iter >= maxiter)
            fprintf('reach maximum iteration.\n')
        end
        if( max_low_bound + eps >= xi)
            fprintf('eps converge.\n');
        end
        fprintf('niter=%d,No cut: %d,upper_bound: %f, lower_bound:%f,gap=%f\n',iter,n_active,xi, max_low_bound,gap);    
        
        break;
    end       
end

history.time = cputime - history.time;


function gamma = get_gamma(loss_type,N,L)

level = get_level(loss_type);

switch(level)
    case 1
        gamma = N*L;
    case 2
        gamma = N;
    case 3
        gamma = L;
    case 4
        gamma = 1;
    otherwise
        error('level is not defined\n');
end


