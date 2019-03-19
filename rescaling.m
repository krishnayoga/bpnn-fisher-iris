%normalisasi rescaling 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fungsinya Xscale = (x- min(x))/(max(x) - min (x)

for m = 1 : length_in_row
    for n = 1 : length_in_col
       mat_in(m,n) = ((mat_in(m,n) - min(mat_in(:,n)))/(max(mat_in(:,n)) - min(mat_in(:,n))));
    end
end
for m = 1 : length_in_row
    for n = 1 : length_in_col
        mat_target(m,n) = ((mat_target(m,n) - min(mat_target(:,n)))/(max(mat_target(:,n)) - min(mat_target(:,n))));
    end
end