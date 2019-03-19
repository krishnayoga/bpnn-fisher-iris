% Backpropagation with XOR data input

% Inisialisasi data
file = 'tes_data.xlsx';
mat_in = xlsread(file,1);
mat_target = xlsread(file,2);
%mat_in = [1 1;1 0;0 1;0 0];
%mat_target = [1; 0; 0; 1];
hidden_n = 2;
target_error = 0.01;

[length_in_row,length_in_col] = size(mat_in);
[length_out_row,length_out_col] = size(mat_target);


% Normalisasi data
% Data input
mat_in = normc(mat_in);
% Data target
mat_target = normc(mat_target);

% Inisialisasi beban dengan Random
weight_hidden_in = rand(length_in_col,hidden_n);
weight_hidden_out = rand(hidden_n,length_out_col);

% Inisialisasi bias
bias_hidden_in = rand(1,hidden_n);
bias_hidden_out = rand(1,length_out_col);

% Inisialisasi matrix update
delta_hidden_in = zeros(length_in_col,hidden_n);
delta_hidden_out = zeros(hidden_n,length_out_col);
delta_bias_hidden_in = zeros(1,hidden_n);
delta_bias_hidden_out = zeros(1,length_out_col);

error_total = 1000;
epoch=0;
alpha = 1;

aaa = 0.5*length_in_row;

while error_total > target_error
    for i = 1:aaa
        for j= 1:hidden_n
            z_in(j) = bias_hidden_in(j) + (mat_in(i,:)*weight_hidden_in(:,j));
            z(j)    = 1/(1+exp(-z_in(j)));
        end
        for k = 1:length_out_col
            y_in(k) = bias_hidden_out(k) + (z*weight_hidden_out(:,k));
            y(k)    = 1/(1+exp(-y_in(k)));
        end
        
        for o = 1:hidden_n
            for l = 1:length_out_col
                d(l) = (mat_target(i,l) - y(l)) * y(l) * (1-y(l));
                delta_hidden_out(o,l) = alpha * d(l) * z(o);
                delta_bias_hidden_out(l) = alpha * d(l);
            end
        end
        
        for m = 1:length_in_col
            for n = 1:hidden_n
                d_in(n) = d(m) * weight_hidden_out(n);
                d(n) = d_in(n) * z(n) * (1-z(n));
                delta_hidden_in(m,n) = alpha * d(n) * mat_in(i,m);
                delta_bias_hidden_in = alpha * d(n);
            end
        end
        
        weight_hidden_in = weight_hidden_in + delta_hidden_in;
        weight_hidden_out= weight_hidden_out + delta_hidden_out;
        bias_hidden_in  = bias_hidden_in + delta_bias_hidden_in;
        bias_hidden_out = bias_hidden_out + delta_bias_hidden_out;
        
        error(i) = 0.5 * (mat_target(i,:)-y) * (mat_target(i,:)-y)';
    end
    
    epoch = epoch + 1;
    error_total(epoch) = sum(error);
end

figure(1)
plot(error_total)
grid
error_final=error_total(epoch); 